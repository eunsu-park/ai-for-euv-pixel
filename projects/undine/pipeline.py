import os
import glob
import asdf
import torch


def load_asdf(file_path):
    af = asdf.open(file_path)
    tree = af.tree
    data = tree['data']
    return data


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, options):
        if options.is_train :
            self.pattern = os.path.join(
                options.data_root, 'train', '*.asdf')
        else :
            self.pattern = os.path.join(
                options.data_root, 'test', '*.asdf')

        self.list_data = sorted(glob.glob(self.pattern))
        self.nb_data = len(self.list_data)
        self.load_fn = load_asdf

    def __len__(self):
        return self.nb_data
    
    def __getitem__(self, idx):
        data = self.load_fn(self.list_data[idx])
        data = torch.tensor(data, dtype=torch.float32)
        return data


class TrainDataset(BaseDataset):
    def __init__(self, options):
        super(TrainDataset, self).__init__(options)


class TestDataset(BaseDataset):
    def __init__(self, options):
        super(TestDataset, self).__init__(options)


if __name__ == "__main__" :

    from options import TrainOptions
    options = TrainOptions().parse()

    dataset = TrainDataset(options)
    print(len(dataset))
    print(dataset[0].shape)
    print(dataset[0].dtype)
    print(dataset[0].device)
    print(dataset[0].requires_grad)
    print(dataset[0].grad)
    print(dataset[0].grad_fn)
    print(dataset[0].is_leaf)
    print(dataset[0].retain_grad)
    print(dataset[0].volatile)