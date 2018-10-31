nvidia-smi
export PYTHONPATH=./:$PYTHONPATH
MODEL=search_ptb
MODEL_DIR=models/$MODEL
LOG_DIR=logs
DATA_DIR=data/penn

mkdir -p $MODEL_DIR/child
mkdir -p $MODEL_DIR/controller
mkdir -p $LOG_DIR

python main_search.py \
  --data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR \
  --child_train_epochs=1000 \
  --child_eval_every_epochs=200 \
  --child_lr=0.25 \
  --child_clip=10.0 \
  --controller_shuffle \
  --controller_encoder_vocab_size=16 \
  --controller_decoder_vocab_size=16 \
  --controller_encoder_emb_size=48 \
  --controller_encoder_hidden_size=96 \
  --controller_decoder_hidden_size=96 \
  --controller_mlp_num_layers=4 \
  --controller_mlp_hidden_size=200 \
  --controller_mlp_dropout=0 \
  --controller_source_length=22 \
  --controller_encoder_length=11 \
  --controller_decoder_length=22 \
  --controller_encoder_dropout=0.1 \
  --controller_decoder_dropout=0 \
  --controller_train_epochs=1000 \
  --controller_optimizer=adam \
  --controller_lr=0.001 \
  --controller_batch_size=100 \
  --controller_trade_off=0.8 \
  --controller_save_frequency=100 2>&1 | tee -a $LOG_DIR/train.$MODEL.log
