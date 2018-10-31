nvidia-smi
export PYTHONPATH=./:$PYTHONPATH
MODEL=ptb
MODEL_DIR=models/$MODEL
LOG_DIR=logs
DATA_DIR=data/penn

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

python test.py \
  --data=$DATA_DIR \
  --save=$MODEL_DIR
