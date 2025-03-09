

nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name default_conv \
    --model_type convolution \
    --device cuda \
    --batch_size 1 \
    & \

nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name default_pixel \
    --model_type pixel \
    --device cuda \
    --batch_size 1 \
    & \


# python train.py \
#     --data_root $HOME/Dataset/pixel \
#     --save_root $HOME/Result/epic \
#     --experiment_name default_pixel \
#     --model_type pixel \
#     --device cuda \
#     --batch_size 1 \


