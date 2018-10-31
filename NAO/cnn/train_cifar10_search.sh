nvidia-smi
MODEL=search_cifar10
MODEL_DIR=models/$MODEL
LOG_DIR=logs
DATA_DIR=data

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

python train_cifar.py \
  --data_dir=$DATA_DIR \
  --dataset=cifar10 \
  --model_dir=$MODEL_DIR \
  --train_epochs=25 \
  --N=3 \
  --filters=32 \
  --num_nodes=5 \
  --drop_path_keep_prob=0.7 \
  --dense_dropout_keep_prob=1.0 \
  --batch_size=128 \
  --epochs_per_eval=1 \
  --lr_max=0.024 \
  --lr_min=0.0 \
  --T_0=100 \
  --T_mul=1 \
  --arch="epd/data/0iter/dag.1.json" \
  --use_aux_head \
  --split_train_valid \
  --num_gpus=1 \
  --lr_schedule=cosine 2>&1 | tee -a $LOG_DIR/train.$MODEL.log
