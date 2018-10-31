nvidia-smi
export PYTHONPATH=./:$PYTHONPATH
MODEL=search_ptb
MODEL_DIR=models
LOG_DIR=logs
DATA_DIR=data/cifar10

mkdir -p $MODEL_DIR/child
mkdir -p $MODEL_DIR/controller
mkdir -p $LOG_DIR

python train_search.py \
  --child_data_format="NCHW" \
  --data_path=$DATA_DIR \
  --output_dir=$MODEL_DIR \
  --child_sample_policy=uniform \
  --child_batch_size=160 \
  --child_num_epochs=150 \
  --child_eval_every_epochs=30 \
  --child_use_aux_heads \
  --child_num_layers=6 \
  --child_out_filters=20 \
  --child_l2_reg=1e-4 \
  --child_num_cells=5 \
  --child_keep_prob=0.90 \
  --child_drop_path_keep_prob=0.60 \
  --child_lr_cosine \
  --child_lr_max=0.05 \
  --child_lr_min=0.0005 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --child_eval_batch_size=500 \
  --controller_encoder_vocab_size=12 \
  --controller_decoder_vocab_size=12 \
  --controller_encoder_emb_size=48 \
  --controller_encoder_hidden_size=96 \
  --controller_decoder_hidden_size=96 \
  --controller_mlp_num_layers=3 \
  --controller_mlp_hidden_size=100 \
  --controller_mlp_dropout=0.1 \
  --controller_source_length=40 \
  --controller_encoder_length=20 \
  --controller_decoder_length=40 \
  --controller_train_epochs=1000 \
  --controller_optimizer=adam \
  --controller_lr=0.001 \
  --controller_batch_size=100 \
  --controller_save_frequency=100 \
  --controller_attention \
  --controller_time_major \
  --controller_symmetry 2>&1 | tee -a $LOG_DIR/train.$MODEL.log
