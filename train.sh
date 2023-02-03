

BATCH_SIZE=1
# 64 oom
BATCH_SIZE=64 
# 32 oom
BATCH_SIZE=32
BATCH_SIZE=24
BATCH_SIZE=4
BATCH_SIZE=8
TOKENS_PER_SAMPLE=512
MAX_TOKEN=$((TOKENS_PER_SAMPLE*BATCH_SIZE))
DATA_DIR=YOUR_DATA_DIR
ARCH=$2
prefix=lm
MAX_UPDATE=50000
WARM_UP=4000
# WARM_UP=2000
UPDATE_FREQ=$(( 128 / $BATCH_SIZE / $1 ))
PORT=$(( $RANDOM + 2000 ))
echo $PORT
# 调整
LR=0.0005
LR=$4
CLIP_NORM=$5
NAME=$3
TYPE=$6
decay=$7

fairseq-train --task language_modeling \
    $DATA_DIR \
    --save-dir checkpoints/$prefix/512_5w_large_decay_${decay}_${ARCH}_${LR} \
    --distributed-world-size $1  --distributed-port $PORT \
    --arch $ARCH --share-decoder-input-output-embed \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay $decay --clip-norm $CLIP_NORM \
    --lr $LR --lr-scheduler inverse_sqrt --warmup-updates $WARM_UP --warmup-init-lr 1e-07 \
    --tokens-per-sample $TOKENS_PER_SAMPLE --sample-break-mode none \
    --max-tokens $MAX_TOKEN --update-freq $UPDATE_FREQ \
    --ddp-backend=legacy_ddp \
    --batch-size $BATCH_SIZE \
    --max-update $MAX_UPDATE --log-interval 10 2>&1 | tee $ARCH.log