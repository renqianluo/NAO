nvidia-smi
MODEL=wt2
MODEL_DIR=models/$MODEL
LOG_DIR=logs
DATA_DIR=data/wikitext-2

fixed_arc="0 1 0 1 1 1 3 0 4 1 4 1 5 1 1 1 7 1 1 1 1 2"

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

python train.py \
  --data=$DATA_DIR \
  --save=$MODEL_DIR \
  --dropouth=0.15 \
  --emsize=700 \
  --nhidlast=700 \
  --nhid=700 \
  --wdecay=5e-7 \
  --epochs=2000 \
  --arch="$fixed_arc" 2>&1 | tee -a $LOG_DIR/train.$MODEL.log
