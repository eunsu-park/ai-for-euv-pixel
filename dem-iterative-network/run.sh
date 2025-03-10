


python train.py \
    --device cpu \
    --data_file_path $HOME/Dataset/pixel/train/aia.euv.2011-01-01-00-00-00.h5 \
    --save_root $HOME/Result/dine \
    --model_type pixel \
    --experiment_name default_pixel \
    --loss_type log_cosh \
    --metric_type mae \
