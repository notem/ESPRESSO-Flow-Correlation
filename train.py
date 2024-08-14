import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math
import os
from os.path import join
import pickle as pkl
from tqdm import tqdm
from torchvision import transforms, utils
import transformers
import scipy
import json
import time
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from utils.nets.espressonet import EspressoNet
from utils.nets.dcfnet import Conv1DModel
from utils.nets.transdfnet import DFNet
from utils.processor import DataProcessor
from utils.data import *
from utils.loss import *



# enable if NaN or other odd behavior appears
#torch.autograd.set_detect_anomaly(True)
# disable any unnecessary logging / debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# auto-optimize cudnn ops
torch.backends.cudnn.benchmark = True



def parse_args():
    parser = argparse.ArgumentParser(
                        prog = 'train.py',
                        description = 'Train a feature extraction network (FEN) for flow correlation.',
                        #epilog = 'Text at the bottom of help'
                        )

    # experiment configuration options
    parser.add_argument('--data_dir', 
                        default = './data/undefended', 
                        type = str,
                        help = "Path to dataset root 'pathx' directory.", 
                        required=True)
    parser.add_argument('--ckpt_dir',
                        default = './exps/ckpts',
                        type = str,
                        help = "Set directory for model checkpoints.")
    parser.add_argument('--log_dir', 
                        default = './exps/logs',
                        type = str,
                        help = "Set directory to store the training history log.")
    parser.add_argument('--ckpt', 
                        default = None, 
                        type = str,
                        help = "Resume from checkpoint path.")
    parser.add_argument('--exp_name',
                        type = str,
                        default = f'{time.strftime("%Y%m%d-%H%M%S")}',
                        help = "")

    # Model architecture options
    parser.add_argument('--config',
                        default = None,
                        type = str,
                        help = "Set model config (as JSON file)")
    parser.add_argument('--input_size', 
                        default = None, 
                        type = int,
                        help = "Overwrite the config .json input length parameter.")
    parser.add_argument('--features', 
                        default=None, type=str, nargs="+",
                        help='Overwrite the features used in the config file. Multiple features can be provided.')
    parser.add_argument('--bs', 
                        default=64, type=int,
                        help='Size of batches.')
    parser.add_argument('--epochs', 
                        default=10000, type=int,
                        help='Number of epochs for training.')
    parser.add_argument('--online', 
                        default=False, action='store_true',
                        help='Use online batch mining strategy.')
    parser.add_argument('--hard', 
                        default=False, action='store_true', 
                        help='Use hard negative mining during training epochs.')
    parser.add_argument('--loss_margin', 
                        default=0.1, type=float, 
                        help='The margin to use for triplet loss and semi-hard mining.')
    parser.add_argument('--single_fen', 
                        default=False, action='store_true',
                        help='Use the same FEN for in and out flows.')
    parser.add_argument('--dcf', 
                        default=False, action='store_true',
                        help='Use the original DeepCoFFEA model and windowing strategy.')
    parser.add_argument('--decay_step',
                        default = 100,
                        type = int,
                        help='Learning rate is decayed after this number of epochs.')

    return parser.parse_args()


if __name__ == "__main__":
    """
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    # load checkpoint (if it exists)
    checkpoint_path = args.ckpt
    checkpoint_fname = None
    resumed = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        resumed = torch.load(checkpoint_path)
        checkpoint_fname = os.path.basename(os.path.dirname(checkpoint_path))
    # else: checkpoint path and fname will be defined later if missing

    checkpoint_dir = args.ckpt_dir
    log_dir = args.log_dir

    # # # # # #
    # finetune config
    # # # # # #
    batch_size      = args.bs
    ckpt_period     = 50
    epochs          = args.epochs
    save_best_epoch = True
    opt_lr          = 1e-3
    opt_betas       = (0.9, 0.999)
    opt_wd          = 0.001
    steplr_step     = args.decay_step
    steplr_gamma    = 0.7
    loss_margin     = args.loss_margin
    use_same_fen    = args.single_fen
    # # # # # #

    # all trainable network parameters
    params = []

    # DF model config
    if resumed:
        model_config = resumed['config']
    elif args.config:
        with open(args.config, 'r') as fi:
            model_config = json.load(fi)
    elif args.dcf:
        model_config = {
                'model': "DCF",
                'feature_dim': 64,
                "features": [
                    "dcf",
                    ],
                "window_kwargs": {
                     'window_count': 11, 
                     'window_width': 5, 
                     'window_overlap': 3,
                     "include_all_window": False,
                    },
            }
    else:
        model_config = {
                'model': "Espresso",
                'input_size': 1200,
                'feature_dim': 64,
                'hidden_dim': 96,
                'depth': 8,
                'input_conv_kwargs': {
                    'kernel_size': 3,
                    'stride': 3,
                    'padding': 0,
                    },
                'output_conv_kwargs': {
                    'kernel_size': 50,
                    'stride': 3,
                    'padding': 0,
                    },
                "mixer_kwargs": {
                    "type": "mhsa",
                    "head_dim": 16,
                    "use_conv_proj": True,
                    "kernel_size": 3,
                    "stride": 2,
                    "feedforward_style": "mlp",
                    "feedforward_ratio": 4,
                    "feedforward_drop": 0.0
                },
                "features": [
                    "interval_dirs_up",
                    "interval_dirs_down",
                    "interval_dirs_sum",
                    "interval_dirs_sub",
                    "interval_size_up",
                    "interval_size_down",
                    "interval_size_sum",
                    "interval_size_sub",
                    "interval_cumul_norm",
                    ],
                "window_kwargs": None,
            }
        
    model_name = model_config['model']
    if args.input_size is not None:
        model_config['input_size'] = args.input_size
    if args.features is not None:
        model_config['features'] = args.features
    features = model_config['features']
    feature_dim = model_config['feature_dim']
    model_arch = model_config['model']

    print("==> Model configuration:")
    print(json.dumps(model_config, indent=4))

    # traffic feature extractor
    if model_arch.lower() == "espresso":
        inflow_fen = EspressoNet(input_channels=len(features),
                                **model_config)
    elif model_arch.lower() == "dcf":
        inflow_fen = Conv1DModel(input_channels=len(features),
                                input_size = 500,
                                **model_config)
    elif model_arch.lower() == "laserbeak":
        inflow_fen = DFNet(input_channels=len(features),
                                input_size = 500,
                                **model_config)
    else:
        import sys
        sys.exit(-1)

    inflow_fen = inflow_fen.to(device)
    if resumed:
        inflow_fen.load_state_dict(resumed['inflow_fen'])
    params += inflow_fen.parameters()

    if use_same_fen:
        outflow_fen = inflow_fen

    else:
        if model_arch.lower() == "espresso":
            # traffic feature extractor
            outflow_fen = EspressoNet(input_channels=len(features),
                                    **model_config)
        elif model_arch.lower() == "dcf":
            outflow_fen = Conv1DModel(input_channels=len(features),
                                    input_size = 800,
                                    **model_config)
        elif model_arch.lower() == "laserbeak":
            outflow_fen = Conv1DModel(input_channels=len(features),
                                    input_size = 800,
                                    **model_config)

        outflow_fen = outflow_fen.to(device)
        if resumed:
            outflow_fen.load_state_dict(resumed['outflow_fen'])
        params += outflow_fen.parameters()

    # # # # # #
    # print parameter count
    param_count = sum(p.numel() for p in params if p.requires_grad)
    param_count /= 1000000
    param_count = round(param_count, 2)
    print(f'=> Model is {param_count}m parameters large.')
    # # # # # #


    # # # # # #
    # create data loaders
    # # # # # #
    # multi-channel feature processor
    processor = DataProcessor(features)

    # dataset split indices
    tr_idx = np.arange(0, 9000)
    va_idx = np.arange(9000, 10000)
    te_idx = np.arange(10000,11000)

    # stream window definitions
    window_kwargs = model_config['window_kwargs']

    def make_dataloader(idx, shuffle=False):
        """
        """
        # load data from files
        samples = load_dataset(args.data_dir, sample_list = idx)

        # build base dataset object
        data = BaseDataset(samples, processor,
                              window_kwargs = window_kwargs,
                          )

        # construct a triplets dataset object, derived from the base dataset object
        if args.online:  # build a dataset compatible for online mining
            dataset = OnlineDataset(data)
        else:            # build a dataset for random triplet generation
            dataset = TripletDataset(data)
        loader = DataLoader(dataset,
                            batch_size = batch_size, 
                            collate_fn = dataset.batchify,
                            shuffle = shuffle)
        return dataset, loader

    # construct dataloaders
    print("Loading testing data...")
    te_data, testloader = make_dataloader(te_idx)
    print("Loading validation data...")
    va_data, validationloader = make_dataloader(va_idx)
    print("Loading training data...")
    tr_data, trainloader = make_dataloader(tr_idx, shuffle=True)


    # # # # # #
    # optimizer and params, reload from resume is possible
    # # # # # #
    optimizer = optim.AdamW(params,
            lr=opt_lr, betas=opt_betas, weight_decay=opt_wd)
    if resumed and resumed.get('opt', None):
        opt_state_dict = resumed['opt']
        optimizer.load_state_dict(opt_state_dict)

    last_epoch = -1
    if resumed and resumed['epoch']:    # if resuming from a finetuning checkpoint
        last_epoch = resumed['epoch']

    # StepLR scheduler for learning rate decay
    scheduler = StepLR(optimizer, 
                        step_size = steplr_step, 
                        gamma = steplr_gamma,
                        last_epoch = last_epoch
                    )

    # define checkpoint fname if not provided
    if not checkpoint_fname:
        checkpoint_fname = f'{model_name}'
        checkpoint_fname += f'_{args.exp_name}'

    # create checkpoint directory if necesary
    if not os.path.exists(f'{checkpoint_dir}/{checkpoint_fname}/'):
        os.makedirs(f'{checkpoint_dir}/{checkpoint_fname}/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    if args.online:
        all_criterion = OnlineCosineTripletLoss(margin=loss_margin)
        hard_criterion = OnlineHardCosineTripletLoss(margin=loss_margin)
    else:
        criterion = CosineTripletLoss(margin=loss_margin)


    def epoch_iter(dataloader, 
                   eval_only=False, 
                   desc=f"Epoch"):
        """Run one epoch over dataset
        """
        tot_loss = 0.
        n = 0
        with tqdm(total=len(dataloader),
                desc = desc,
                dynamic_ncols = True) as pbar:

            for batch_idx, data in enumerate(dataloader):

                # dataloader returns batches of PxK samples, and need to build triplets
                if args.online:
                    inflow, outflow = data
                    inflow = inflow.to(device)
                    outflow = outflow.to(device)

                    inflow_embed = inflow_fen(inflow)
                    outflow_embed = outflow_fen(outflow)

                    if args.hard:
                        loss = hard_criterion(inflow_embed, outflow_embed)
                    else:
                        loss = all_criterion(inflow_embed, outflow_embed)

                    n += len(inflow)

                # dataloader returns already formed triplets
                else:
                    inputs_anc, inputs_pos, inputs_neg = data
                    inputs_anc = inputs_anc.to(device)
                    inputs_pos = inputs_pos.to(device)
                    inputs_neg = inputs_neg.to(device)

                    # # # # # #
                    # generate traffic feature vectors & run triplet loss
                    anc_embed = inflow_fen(inputs_anc)
                    pos_embed = outflow_fen(inputs_pos)
                    neg_embed = outflow_fen(inputs_neg)
                    loss = criterion(anc_embed, pos_embed, neg_embed)

                    n += len(inputs_anc)

                if not eval_only:
                    loss.backward()
                    optimizer.step()
                    for param in params:
                        param.grad = None

                tot_loss += loss.item()
                
                pbar.update(1)
                pbar.set_postfix({
                                  'triplet': tot_loss/(batch_idx+1),
                                  })
                
                last_lr = scheduler.get_last_lr()
                if last_lr and not eval_only:
                    mod_desc = desc + f'[lr={last_lr[0]}]'
                else:
                    mod_desc = desc
                pbar.set_description(mod_desc)

        tot_loss /= batch_idx + 1
        return tot_loss


    # do training
    history = {}
    try:
        for epoch in range(last_epoch+1, epochs):

            # train and update model using training data
            inflow_fen.train()
            outflow_fen.train()
            train_loss = epoch_iter(trainloader, 
                                    desc = f"Epoch {epoch} Train")
            scheduler.step()
            metrics = {'tr_loss': train_loss}

            # evaluate on hold-out data
            inflow_fen.eval()
            outflow_fen.eval()
            if validationloader is not None:
                with torch.no_grad():
                    va_loss = epoch_iter(validationloader, 
                                            eval_only = True, 
                                            desc = f"Epoch {epoch} Val.")
                metrics.update({'va_loss': va_loss})
            with torch.no_grad():
                test_loss = epoch_iter(testloader, 
                                        eval_only = True, 
                                        desc = f"Epoch {epoch} Test")
            metrics.update({'te_loss': test_loss})

            # save model
            if (epoch % ckpt_period) == (ckpt_period-1):
                # save last checkpoint before restart
                checkpoint_path_epoch = f"{checkpoint_dir}/{checkpoint_fname}/e{epoch}.pth"
                print(f"Saving end-of-cycle checkpoint to {checkpoint_path_epoch}...")
                torch.save({
                                "epoch": epoch,
                                "inflow_fen": inflow_fen.state_dict(),
                                "outflow_fen": outflow_fen.state_dict(),
                                "opt": optimizer.state_dict(),
                                "config": model_config,
                        }, checkpoint_path_epoch)

            if save_best_epoch:
                best_val_loss = min([999]+[metrics['va_loss'] for metrics in history.values()])
                if metrics['va_loss'] < best_val_loss:
                    checkpoint_path_epoch = f"{checkpoint_dir}/{checkpoint_fname}/best.pth"
                    print(f"Saving new best model to {checkpoint_path_epoch}...")
                    torch.save({
                                    "epoch": epoch,
                                    "inflow_fen": inflow_fen.state_dict(),
                                    "outflow_fen": outflow_fen.state_dict(),
                                    "opt": optimizer.state_dict(),
                                    "config": model_config,
                            }, checkpoint_path_epoch)

            history[epoch] = metrics

            if not args.online:  # generate new triplets
                tr_data.generate_triplets(fens = (inflow_fen, outflow_fen), 
                                          margin = loss_margin if not args.hard else 0.)

    except KeyboardInterrupt:
        pass

    finally:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        results_fp = f'{log_dir}/{checkpoint_fname}.txt'
        with open(results_fp, 'w') as fi:
            json.dump(history, fi, indent='\t')

