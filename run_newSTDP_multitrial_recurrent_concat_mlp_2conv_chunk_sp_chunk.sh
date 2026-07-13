CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 main_newSTDP_multitrial_recurrent_concat_mlp_2conv_chunk_sp_chunk.py \
  --distributed \
  --exp_name recurrent_edge_seq400_history200_conv20_stride10_dilated_gru_chunk20_spchunk2 \
  --train_list \
    data/data10/spike_seed6_300k.npy \
    data/data10/spike_seed16_300k.npy \
    data/data10/spike_seed30_300k.npy \
    data/data10/spike_seed37_300k.npy \
    data/data10/spike_seed73_300k.npy \
    data/data10/spike_seed131_300k.npy \
    data/data10/spike_seed133_300k.npy \
    data/data10/spike_seed134_300k.npy \
  --val_list \
    data/data10/spike_seed142_300k.npy \
  --test_list \
    data/data10/spike_seed143_300k.npy \
  --totalsteps 300000 \
  --skipsteps 0 \
  --seq_len 400 \
  --seq_stride 200 \
  --history 200 \
  --conv_filter_size 20 \
  --local_conv_stride 10 \
  --conv2_time_kernel 19 \
  --conv_stride 1 \
  --embed_mode none \
  --posemb_dim 0 \
  --edge_chunk_num 20 \
  --out_channel 32 \
  --n_hid 32 \
  --gru_hidden 32 \
  --gru_layers 1 \
  --batch_size 8 \
  --lr 5e-4 \
  --max_epoch 300 \
  --use_wandb

#   CUDA_VISIBLE_DEVICES=4,5,6,7 \
# torchrun --standalone --nproc_per_node=4 main_newSTDP_multitrial_recurrent_concat_mlp.py \
#   --distributed \
#   --exp_name recurrent_edge_seq250_history200_noPE_concat_conv200_lr5e-4\
#   --train_list \
#     data/data10/spike_seed6_300k.npy \
#     data/data10/spike_seed16_300k.npy \
#     data/data10/spike_seed30_300k.npy \
#     data/data10/spike_seed37_300k.npy \
#     data/data10/spike_seed73_300k.npy \
#     data/data10/spike_seed131_300k.npy \
#     data/data10/spike_seed133_300k.npy \
#     data/data10/spike_seed134_300k.npy \
#   --val_list \
#     data/data10/spike_seed142_300k.npy \
#   --test_list \
#     data/data10/spike_seed143_300k.npy \
#   --totalsteps 300000 \
#   --skipsteps 0 \
#   --seq_len 250 \
#   --seq_stride 50 \
#   --history 200 \
#   --conv_filter_size 200 \
#   --conv_stride 1 \
#   --embed_mode none \
#   --posemb_dim 0 \
#   --out_channel 32 \
#   --gru_hidden 32 \
#   --gru_layers 1 \
#   --batch_size 8 \
#   --lr 5e-4 \
#   --max_epoch 300 \
#   --patience 300 \
#   --beta_l2 0.0 \
#   --use_wandb