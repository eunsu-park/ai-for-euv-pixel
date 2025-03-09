

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

nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name sigmoid_conv \
    --model_type convolution \
    --device cuda \
    --batch_size 1 \
    --latent_output_type sigmoid \
    & \

nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name sigmoid_pixel \
    --model_type pixel \
    --device cuda \
    --batch_size 1 \
    --latent_output_type sigmoid \
    & \

nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name softmax_conv \
    --model_type convolution \
    --device cuda \
    --batch_size 1 \
    --latent_output_type softmax \
    & \

nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name softmax_pixel \
    --model_type pixel \
    --device cuda \
    --batch_size 1 \
    --latent_output_type softmax \
    & \


# python train.py \
#     --data_root $HOME/Dataset/pixel \
#     --save_root $HOME/Result/epic \
#     --experiment_name default_pixel \
#     --model_type pixel \
#     --device cuda \
#     --batch_size 1 \


