#! /usr/bin/bash

batch_size=2
data_dir=path_to_bin_data
ARCH=$1
ckpt=your_ckpt_dir/$ARCH/checkpoint_best.pt
l=$2

fairseq-eval-lm \
    $data_dir \
    --sample-break-mode none \
    --path $ckpt \
    --max-sentences 1 \
    --model-overrides \"{'max_tokens':$l, 'tokens_per_sample':$l, 'max_target_positions':$l}\"
    --max-tokens $l \
    --tokens-per-sample $l \
    --max-target-positions $l \
    --context-window 0