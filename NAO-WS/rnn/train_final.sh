nvidia-smi
export PYTHONPATH=./:$PYTHONPATH
MODEL_DIR=models/
LOG_DIR=logs
DATA_DIR=data/penn

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

fixed_arc="0 1 1 1 2 2 1 2 4 1 4 1 4 1 7 1 8 2 1 0 4 1"

python train.py \
  --data=$DATA_DIR \
  --save=$MODEL_DIR \
  --epochs=4000 \
  --arch="$fixed_arc" 2>&1 | tee -a $LOG_DIR/train.log
