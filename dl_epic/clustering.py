import os
from glob import glob
import asdf
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import pickle
import random
import time
import logging
import torch
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--num_clusters", type=int, default=5)

options = parser.parse_args()


NUM_CLUSTERS = options.num_clusters  # 클러스터 개수
DATA_ROOT = "/home/eunsu/Dataset/epic"  # 데이터 파일이 저장된 디렉토리
SAVE_ROOT = f"/storage/aisolo/epic/clustering_{NUM_CLUSTERS}"  # 결과 파일이 저장될 디렉토리
LOG_PATH = f"{SAVE_ROOT}/log.txt"  # 로그 파일 경로
MODEL_DIR = f"{SAVE_ROOT}/model"  # 모델 파일이 저장될 디렉토리
TEST_DIR = f"{SAVE_ROOT}/test"    # 테스트 결과 파일이 저장될 디렉토리
MAX_ITER = 300                  # 클러스터링 최대 반복 횟수
NUM_EPOCHS = 10                 # 클러스터링 학습 횟수
BATCH_SIZE = 1024*1024          # 클러스터링 학습 배치 크기
MODEL_SAVE_FREQ = 1            # 모델 저장 주기

if not os.path.exists(SAVE_ROOT):
    os.makedirs(SAVE_ROOT)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)

def read_data(file_path):
    af = asdf.open(file_path)
    tree = af.tree
    data = tree["data"]
    data = np.transpose(data, (1, 2, 0))
    data = data.reshape(-1, 6)
    return data


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        list_data = glob(f"{DATA_ROOT}/train/*.asdf")
        random.shuffle(list_data)
        self.list_data = list_data

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, index):
        file_path = self.list_data[index]
        data = read_data(file_path)
        return data


if __name__ == "__main__" :

    logger = logging.getLogger("clustering")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH)
    logger.addHandler(handler)
    logger.info("start")

    model = MiniBatchKMeans(
        n_clusters=NUM_CLUSTERS,
        max_iter=MAX_ITER,
        batch_size=BATCH_SIZE
    )

    dataset = BaseDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        shuffle=True, num_workers=8)
    # list_data = glob(f"{DATA_ROOT}/train/*.asdf")
    # random.shuffle(list_data)
    # list_data = list_data

    n = 0
    epoch = 0
    t0 = time.time()
    while epoch < NUM_EPOCHS:
        message = f"epoch: {epoch}"
        logger.info(message)
        print(message)
        for _, data in enumerate(dataloader):
            data = np.squeeze(data.numpy())
            model.partial_fit(data)
            n += 1
            if n % 1000 == 0:
                message = f"Epoch:{epoch:2d} Iteration:{n:7d} {time.time()-t0:.2f} sec"
                logger.info(message)
                print(message)
                t0 = time.time()
        epoch += 1
        if epoch % MODEL_SAVE_FREQ == 0:
            model_path = os.path.join(MODEL_DIR, f"model_{epoch:04d}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            message = f"save: {model_path}"
            logger.info(message)
            print(message)

    message = f"fitting done"
    logger.info(message)
    print(message)

    model_path = os.path.join(SAVE_ROOT, f"model_final.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    message = f"save: {model_path}"
    logger.info(message)
    print(message)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    message = f"load: {model_path}"
    logger.info(message)
    print(message)    

    list_data = sorted(glob(f"{DATA_ROOT}/test/*.asdf"))

    for file_path in list_data :

        file_name = os.path.basename(file_path)
        date = file_name.split(".")[2]
        save_dir = f"{TEST_DIR}/{date}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data = read_data(file_path)
        result = model.predict(data).reshape(1024, 1024)
        data = data.reshape(1024, 1024, 6)

        np.savez(f"{save_dir}/result.npz", data=data, result=result)

        image_94 = data[..., 0]#.reshape(1024, 1024)
        image_131 = data[..., 1]#.reshape(1024, 1024)
        image_171 = data[..., 2]#.reshape(1024, 1024)
        image_193 = data[..., 3]#.reshape(1024, 1024)
        image_211 = data[..., 4]#.reshape(1024, 1024)
        image_335 = data[..., 5]#.reshape(1024, 1024)

        for n in range(NUM_CLUSTERS):

            save_name = f"cluster_{date}_{n}"

            mask = result == n
            masked_94 = image_94.copy()
            masked_131 = image_131.copy()
            masked_171 = image_171.copy()
            masked_193 = image_193.copy()
            masked_211 = image_211.copy()
            masked_335 = image_335.copy()

            masked_94[mask] = np.nan
            masked_131[mask] = np.nan
            masked_171[mask] = np.nan
            masked_193[mask] = np.nan
            masked_211[mask] = np.nan
            masked_335[mask] = np.nan

            plt.figure(figsize=(60, 20))
            plt.subplot(2, 6, 1)
            plt.imshow(image_94, vmin=-1, vmax=1, cmap="gray")

            plt.subplot(2, 6, 2)
            plt.imshow(image_131, vmin=-1, vmax=1, cmap="gray")

            plt.subplot(2, 6, 3)
            plt.imshow(image_171, vmin=-1, vmax=1, cmap="gray")

            plt.subplot(2, 6, 4)
            plt.imshow(image_193, vmin=-1, vmax=1, cmap="gray")

            plt.subplot(2, 6, 5)
            plt.imshow(image_211, vmin=-1, vmax=1, cmap="gray")

            plt.subplot(2, 6, 6)
            plt.imshow(image_335, vmin=-1, vmax=1, cmap="gray")

            plt.subplot(2, 6, 7)
            plt.imshow(masked_94, vmin=-1, vmax=1, cmap="gray")

            plt.subplot(2, 6, 8)
            plt.imshow(masked_131, vmin=-1, vmax=1, cmap="gray")

            plt.subplot(2, 6, 9)
            plt.imshow(masked_171, vmin=-1, vmax=1, cmap="gray")

            plt.subplot(2, 6, 10)
            plt.imshow(masked_193, vmin=-1, vmax=1, cmap="gray")

            plt.subplot(2, 6, 11)
            plt.imshow(masked_211, vmin=-1, vmax=1, cmap="gray")

            plt.subplot(2, 6, 12)
            plt.imshow(masked_335, vmin=-1, vmax=1, cmap="gray")

            plt.savefig(f"{save_dir}/{save_name}.png", dpi=100)
            plt.close()




