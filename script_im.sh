GPUS=16
BATCH=128
# you can also choose tno_vit_e3g1_small_rpe_l1_90_prenorm
arch=tnn_vit_e3g1_tiny_rpe_l1_90_prenorm
LR=0.0005
DATASET=IMNET
decay=0.05
warmup=10
clip_grad=5

bash train_im.sh GPUS BATCH arch \
                 arch LR DATASET \
                 decay warmup clip_grad