
##################
# Model architecture and dataset configuration
# !! MODIFY THIS
#
MODEL_ARCH=hotwater     # espresso w/ no mixer
MODEL_ARCH=greentea     # espresso w/ conv. mixer
MODEL_ARCH=dcf          # dfnet
MODEL_ARCH=espresso     # espresso w/ mhsa mixer
ARCH_CONFIG=./configs/nets/${MODEL_ARCH}.json

EXP_CONFIG=./configs/exps/obfs4.json
EXP_CONFIG=./configs/exps/august.json
EXP_CONFIG=./configs/exps/june.json
#
#########


##################
# setup extra training script args
# !! MODIFY THIS
#
TRAIN_MODE=online           # 'online' or anything else (impacts directory name)
TRAIN_MODE=offline
EXTRA="--margin 0.1"        # setup loss margin arg
#EXTRA="--margin 0.5"        # setup loss margin arg
#EXTRA="--margin 1.0"        # setup loss margin arg

# add extra arguments
#EXTRA="$EXTRA --decay_step 200" 
#EXTRA="$EXTRA --decay_step 50" 
#EXTRA="$EXTRA --bs 256" 
#EXTRA="$EXTRA --epochs 1000" 
#EXTRA="$EXTRA --single_fen" 
#
#########################

if [ "$TRAIN_MODE" == "online" ]; then
    EXTRA="$EXTRA --online --hard"
fi


##################
# subpath to model file (after training)
# !! MODIFY THIS
#
TRAINED_NET=DCF_20240821-145032       # margin 0.1, offline
CKPT_NAME=e1599.pth
TRAINED_NET=Espresso_20240821-163309  # margin 0.5, online
CKPT_NAME=e2099.pth
TRAINED_NET=Espresso_20240821-163510  # margin 0.1, online
CKPT_NAME=e2049.pth
TRAINED_NET=Espresso_20240823-143233  # margin 1.0 online
CKPT_NAME=e2299.pth
TRAINED_NET=Espresso_20240823-150648  # hotwater 0.5 online
CKPT_NAME=e6249.pth
TRAINED_NET=Espresso_20240823-150609  # greentea 0.5 online
CKPT_NAME=e4649.pth
TRAINED_NET=Espresso_20240821-153443  # margin 0.1, offline
CKPT_NAME=e749.pth
#
##################


##################
#  Script Toggles
#
TRAIN=true     # train FEN model w/ triplet learning
SIMS=false      # generate sim. matrix using FEN
MLP=false        # eval. using MLP classifier
THR=false        # eval. using local thresholds
#
#################


#
# Directory and file definitions
#
EXPNAME=${MODEL_ARCH}_${TRAIN_MODE}
CKPT_DIR=./exps/$EXPNAME/ckpts
LOG_DIR=./exps/$EXPNAME/log
DISTS_FILE=./exps/$EXPNAME/sims/${TRAINED_NET}.pkl
RES1_FILE=./exps/$EXPNAME/mlp/${TRAINED_NET}/metrics.pkl
RES2_FILE=./exps/$EXPNAME/thr/${TRAINED_NET}/metrics.pkl


#
# train a flow correlation FEN
#
if $TRAIN; then
    python src/train.py \
        --exp_config $EXP_CONFIG \
        --net_config $ARCH_CONFIG \
        --ckpt_dir $CKPT_DIR \
        --log_dir $LOG_DIR $EXTRA
fi

#
# create similarity matrix using trained FENs
#
if $SIMS; then
    python src/calc-sims.py \
        --exp_config $EXP_CONFIG \
        --dists_file $DISTS_FILE \
        --ckpt $CKPT_DIR/$TRAINED_NET/$CKPT_NAME
fi

#
# evaluate performance using a trained MLP
#
if $MLP; then
    if [ "$MODEL_ARCH" == "dcf" ]; then
        DROP_RATE=0.3
    else
        DROP_RATE=0.7
    fi
    python src/benchmark-mlp.py \
        --dists_file $DISTS_FILE \
        --results_file $RES1_FILE \
        --dropout $DROP_RATE
fi

#
# evaluate performance using local thresholding
#
if $THR; then
    python src/benchmark-thr.py \
        --dists_file $DISTS_FILE \
        --results_file $RES2_FILE
fi

