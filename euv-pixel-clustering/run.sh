
nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name ae_pixel_2_128 \
    --network_type autoencoder \
    --layer_type pixel \
    --num_latent_features 2 \
    --num_hidden_features 128 \
    --device cuda \
    > ae_pixel_2_128.log 2>&1 & \

nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name ae_pixel_2_1024 \
    --network_type autoencoder \
    --layer_type pixel \
    --num_latent_features 2 \
    --num_hidden_features 1024 \
    --device cuda \
    > ae_pixel_2_1024.log 2>&1 & \


nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name ae_pixel_50_128 \
    --network_type autoencoder \
    --layer_type pixel \
    --num_latent_features 50 \
    --num_hidden_features 128 \
    --device cuda \
    > ae_pixel_50_128.log 2>&1 & \

nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name ae_pixel_50_1024 \
    --network_type autoencoder \
    --layer_type pixel \
    --num_latent_features 50 \
    --num_hidden_features 1024 \
    --device cuda \
    > ae_pixel_50_1024.log 2>&1 & \


nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name vae_pixel_2_128 \
    --network_type variational_autoencoder \
    --layer_type pixel \
    --num_latent_features 2 \
    --num_hidden_features 128 \
    --device cuda \
    > vae_pixel_2_128.log 2>&1 & \

nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name vae_pixel_2_1024 \
    --network_type variational_autoencoder \
    --layer_type pixel \
    --num_latent_features 2 \
    --num_hidden_features 1024 \
    --device cuda \
    > vae_pixel_2_1024.log 2>&1 & \

nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name vae_pixel_50_128 \
    --network_type variational_autoencoder \
    --layer_type pixel \
    --num_latent_features 50 \
    --num_hidden_features 128 \
    --device cuda \
    > vae_pixel_50_128.log 2>&1 & \

nohup python train.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name vae_pixel_50_1024 \
    --network_type variational_autoencoder \
    --layer_type pixel \
    --num_latent_features 50 \
    --num_hidden_features 1024 \
    --device cuda \
    > vae_pixel_50_1024.log 2>&1 & \
