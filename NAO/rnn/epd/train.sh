SETTING=0iter
MODEL_DIR=models/$SETTING
LOG_DIR=logs
DATA_DIR=data/$SETTING
mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

python main.py \
  --mode=train \
  --data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR \
  --encoder_num_layers=1 \
  --encoder_emb_size=96 \
  --encoder_hidden_size=96 \
  --mlp_num_layers=3 \
  --mlp_hidden_size=200 \
  --decoder_num_layers=1 \
  --decoder_hidden_size=96 \
  --source_length=22 \
  --encoder_length=22 \
  --decoder_length=22 \
  --encoder_vocab_size=16 \
  --decoder_vocab_size=16 \
  --encoder_dropout=0.1 \
  --decoder_dropout=0 \
  --mlp_dropout=0.1 \
  --train_epochs=1000 \
  --eval_frequency=10 \
  --batch_size=50 \
  --trade_off=0.8 \
  --time_major \
  --optimizer=adam \
  --attention \
  --lr=0.001 2>&1 | tee $LOG_DIR/train.$SETTING.log
