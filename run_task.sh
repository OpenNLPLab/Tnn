export HYDRA_FULL_ERROR=1
export DATA_PATH=path_to_your_lra_dir

program_path=path_to_your_lra_dir

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
model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} \
model.norm=${NORM} model.prenorm=${PRENORM} train.seed=2222 \
model.gtu_rpe_dim=${GTU_DPB_DIM} \
model.expand_ratio_gtu=${EXPAND_RATIO_GTU} model.expand_ratio_glu=${EXPAND_RATIO_GLU} \
model.gtu_use_decay=True model.gtu_gamma=${GTU_GAMMA} \
model.dropout=${dropout}
