MODEL_DIR=models
LOG_DIR=logs
DATA_DIR=data
LAMBDA=$1
mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

python main.py \
  --mode=predict \
  --data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR \
  --batch_size=50 \
  --time_major \
  --attention \
  --predict_lambda=$LAMBDA
  --predict_from_file=$DATA_DIR/top100 \
  --predict_to_file=$DATA_DIR/top100.lambda$LAMBDA 2>&1 | tee $LOG_DIR/train.log

