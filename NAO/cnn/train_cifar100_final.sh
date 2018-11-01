nvidia-smi
MODEL=cifar100
MODEL_DIR=models/$MODEL
LOG_DIR=logs
DATA_DIR=data

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

python train_cifar.py \
  --data_dir=$DATA_DIR \
  --dataset=cifar100 \
  --model_dir=$MODEL_DIR \
  --train_epochs=600 \
  --N=6 \
  --filters=128 \
  --num_nodes=5 \
  --drop_path_keep_prob=0.6 \
  --dense_dropout_keep_prob=0.8 \
  --batch_size=128 \
  --epochs_per_eval=1 \
  --lr_max=0.024 \
  --lr_min=0.0 \
  --T_0=600 \
  --T_mul=1 \
  --arch="NAONet" \
  --use_aux_head \
  --num_gpus=2 \
  --cutout_size=16 \
  --lr_schedule=cosine 2>&1 | tee -a $LOG_DIR/train.$MODEL.log
