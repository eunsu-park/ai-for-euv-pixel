
python network.py \
    --data_root $HOME/Dataset/pixel \
    --save_root $HOME/Result/epic \
    --experiment_name ae_pixel_2_128 \
    --network_type autoencoder \
    --layer_type pixel \
    --num_latent_features 2 \
    --num_hidden_features 128 \
    --device cuda \
