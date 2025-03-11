
nohup python networks.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name ae_pixel_high \
    --network_type autoencoder \
    --layer_type pixel \
    --num_latent_features 2 \
    --device cuda \
    & \

nohup python networks.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name ae_pixel_high \
    --network_type variational_autoencoder \
    --layer_type pixel \
    --num_latent_features 2 \
    --device cuda \
    & \