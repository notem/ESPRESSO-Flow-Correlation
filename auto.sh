DATA_DIR=./data/undefended
ARCH_CONFIG=./configs/decaf-espresso.json
#EXTRA="--dcf --loss_margin 0.1 --decay_step 1000 --epochs 100000"
EXTRA="--loss_margin 0.1 --online --hard"
# name for directory paths
EXPNAME=decaf_a   # needs to be unique

# Directory and file definitions
CKPT_DIR=./exps/$EXPNAME/ckpts
LOG_DIR=./exps/$EXPNAME/log
DISTS_FILE=./exps/$EXPNAME/dists.pkl
RES1_FILE=./exps/$EXPNAME/mlp/res.pkl
RES2_FILE=./exps/$EXPNAME/thr/res.pkl

# train a flow correlation FEN
python train.py --data_dir $DATA_DIR \
    --config $ARCH_CONFIG \
    --ckpt_dir $CKPT_DIR \
    --log_dir $LOG_DIR $EXTRA

# evaluate FEN with 2nd-stage
#python calc-sims.py --data_dir $DATA_DIR \
#    --dists_file $DISTS_FILE
#python benchmark-mlp.py --dists_file $DISTS_FILE \
#    --results_file $RES1_FILE
#python benchmark-thr.py --dists_file $DISTS_FILE \
#    --results_file $RES2_FILE
