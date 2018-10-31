nvidia-smi
export PYTHONPATH=./:$PYTHONPATH
MODEL_DIR=models
LOG_DIR=logs
DATA_DIR=data/cifar10

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

fixed_arc="0 1 0 4 2 2 1 3 1 0 2 0 2 3 1 3 1 2 0 4 1 1 0 3 2 3 2 3 1 0 0 1 0 4 2 4 2 3 2 3"

python test.py \
  --data_path="$DATA_DIR" \
  --output_dir="$MODEL_DIR" \
  --child_data_format="NCHW" \
  --child_batch_size=144 \
  --child_num_epochs=630 \
  --child_eval_every_epochs=1 \
  --child_fixed_arc="$fixed_arc" \
  --child_use_aux_heads \
  --child_num_layers=15 \
  --child_out_filters=36 \
  --child_num_branches=5 \
  --child_num_cells=5 \
  --child_keep_prob=0.8 \
  --child_drop_path_keep_prob=0.6 \
  --child_l2_reg=2e-4 \
  --child_lr_cosine \
  --child_lr_max=0.05 \
  --child_lr_min=0.0001 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2
