
#MODEL_ARCH=espresso
#MODEL_ARCH=greentea
#MODEL_ARCH=hotwater
MODEL_ARCH=dcf

EXP_CONFIG=./configs/exps/june.json
#EXP_CONFIG=./configs/exps/august.json
#EXP_CONFIG=./configs/exps/obfs4.json

ARCH_CONFIG=./configs/nets/${MODEL_ARCH}.json

TRAIN_MODE=offline
EXTRA="--margin 0.1"

if [ "$TRAIN_MODE" == "online" ]; then
    EXTRA="$EXTRA --online --hard"
fi
if [ "$MODEL_ARCH" == "dcf" ]; then
    EXTRA="$EXTRA --dcf"
fi

# Directory and file definitions
EXPNAME=${MODEL_ARCH}_${TRAIN_MODE}
CKPT_DIR=./exps/$EXPNAME/ckpts
LOG_DIR=./exps/$EXPNAME/log
DISTS_FILE=./exps/$EXPNAME/dists.pkl
RES1_FILE=./exps/$EXPNAME/mlp/res.pkl
RES2_FILE=./exps/$EXPNAME/thr/res.pkl

# subpath to model file (after training)
TRAINED_NET=DCF_20240820-223613/e249.pth

TRAIN=true
SIMS=false
MLP=false
THR=false


# train a flow correlation FEN
if $TRAIN; then
    python train.py \
        --exp_config $EXP_CONFIG \
        --net_config $ARCH_CONFIG \
        --ckpt_dir $CKPT_DIR \
        --log_dir $LOG_DIR $EXTRA
fi

# create similarity matrix using trained FENs
if $SIMS; then
    python calc-sims.py \
        --exp_config $EXP_CONFIG \
        --dists_file $DISTS_FILE \
        --ckpt $CKPT_DIR/$TRAINED_NET
fi

# evaluate performance using a trained MLP
if $MLP; then
    if [ "$MODEL_ARCH" == "dcf" ]; then
        DROP_RATE=0.3
    else
        DROP_RATE=0.7
    fi
    python benchmark-mlp.py \
        --dists_file $DISTS_FILE \
        --results_file $RES1_FILE \
        --drop $DROP_RATE
fi

# evaluate performance using local thresholding
if $THR; then
    python benchmark-thr.py \
        --dists_file $DISTS_FILE \
        --results_file $RES2_FILE
fi

