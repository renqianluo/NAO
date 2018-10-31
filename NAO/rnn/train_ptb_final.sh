nvidia-smi
MODEL=ptb
MODEL_DIR=models/$MODEL
LOG_DIR=logs
DATA_DIR=data/penn

fixed_arc="0 1 0 1 1 1 3 0 4 1 4 1 5 1 1 1 7 1 1 1 1 2"

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

python train.py \
  --data=$DATA_DIR \
  --save=$MODEL_DIR \
  --epochs=4000 \
  --arch="$fixed_arc" 2>&1 | tee -a $LOG_DIR/train.$MODEL.log
