export HYDRA_FULL_ERROR=1
export DATA_PATH=PATH_TO_LRA_DATA
program_path=PATH_TO_LRA_DIR

TASK=$1
ARCH=$2
BS=$3
N_LAYERS=$4
D_MODEL=$5
GTU_DPB_DIM=$6
NORM=$7
EXPAND_RATIO_GTU=$8
EXPAND_RATIO_GLU=$9
GTU_USE_DECAY=${10}
GTU_GAMMA=${11}
lr=${12}
wd=${13}
cards=${14}
n_works=${15}
dropout=${16}
n_works=4
dpb_type=${17}
dpb_layers=${18}
PRENORM=${19}
warmup_steps=${20}

python ${program_path}/train.py wandb=null experiment=${ARCH}-lra-${TASK} \
trainer.gpus=$cards \
loader.batch_size=${BS} \
loader.num_workers=${n_works} \
scheduler.num_warmup_steps=${warmup_steps} \
optimizer.lr=${lr} optimizer.weight_decay=${wd} \
model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.tno_dpb_dim=${GTU_DPB_DIM} \
model.norm=${NORM} model.prenorm=${PRENORM} train.seed=2222 \
model.expand_ratio_tno=${EXPAND_RATIO_GTU} model.expand_ratio_glu=${EXPAND_RATIO_GLU} \
model.tno_use_decay=True model.tno_gamma=${GTU_GAMMA} \
model.dropout=${dropout} model.dpb_type=${dpb_type} model.dpb_layers=${dpb_layers}
