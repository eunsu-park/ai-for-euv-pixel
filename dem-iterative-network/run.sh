
nohup python train.py \
    --device cuda \
    --data_file_path $HOME/Dataset/pixel/train/aia.euv.2011-01-01-00-00-00.h5 \
    --save_root $HOME/Result/dine \
    --model_type pixel \
    --experiment_name pixel_logcos \
    --loss_type log_cosh \
    --metric_type mae \
    & \


nohup python train.py \
    --device cuda \
    --data_file_path $HOME/Dataset/pixel/train/aia.euv.2011-01-01-00-00-00.h5 \
    --save_root $HOME/Result/dine \
    --model_type conv \
    --experiment_name conv_logcosh \
    --loss_type log_cosh \
    --metric_type mae \
    & \
