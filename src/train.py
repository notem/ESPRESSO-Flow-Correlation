import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import json
import time
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import sys

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
                        description = 'Train a feature extraction network (FEN) for flow correlation.\
                            Includes support for training FENs using DeepCoFFEA and ESPRESSO methods.',
                        epilog = '!! This is research-tier code. YMMV'
                        )

    # experiment configuration options
    parser.add_argument('--exp_config',
                        default = './configs/exps/june.json',
                        type = str,
                        help = "Set dataset configuration (as JSON file).", 
                        required = True)
    parser.add_argument('--ckpt_dir',
                        default = './exps/ckpts',
                        type = str,
                        help = "Set directory for model checkpoints.")
    parser.add_argument('--ckpt_name',
                        default=None,
                        type = str,
                        help = "Set the name used for the checkpoint directory.")
    parser.add_argument('--log_dir', 
                        default = './exps/logs',
                        type = str,
                        help = "Set directory to store the training history log.")
    parser.add_argument('--resume', 
                        default = None, 
                        type = str,
                        help = "Resume from checkpoint path.")

    # Model architecture options
    parser.add_argument('--net_config',
                        default = './configs/nets/espresso.json',
                        type = str,
                        help = "Set model config (as JSON file)",
                        required = True)
    parser.add_argument('--input_size', 
                        default = None, 
                        type = int,
                        help = "Overwrite the config .json input length parameter.")
    parser.add_argument('--features', 
                        default=None, type=str, nargs="+",
                        help='Overwrite the features used in the config file. Multiple features can be provided.')
    parser.add_argument('--wd', 
                        default=1e-2, type=float,
                        help='Magnitude of weight decay.')
    parser.add_argument('--bs', 
                        default=128, type=int,
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
    parser.add_argument('--margin', 
                        default=0.1, type=float, 
                        help='The margin to use for triplet loss and semi-hard mining.')
    parser.add_argument('--single_fen', 
                        default=False, action='store_true',
                        help='Use the same FEN for in and out flows.')
    parser.add_argument('--softer_loss', 
                        default=False, action='store_true',
                        help='Include zero-loss samples when taking average batch loss.')
    parser.add_argument('--decay_step',
                        default = 100,
                        type = int,
                        help='Learning rate is decayed by x0.7 after this number of epochs.')

    return parser.parse_args()


if __name__ == "__main__":
    """
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    # load checkpoint (if it exists)
    checkpoint_fname = args.ckpt_name
    checkpoint_path = args.resume
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
    ckpt_period     = min(50, args.decay_step / 2)
    epochs          = args.epochs
    opt_lr          = 1e-3
    opt_betas       = (0.9, 0.999)
    opt_wd          = args.wd
    steplr_step     = args.decay_step
    steplr_gamma    = 0.7
    loss_margin     = args.margin
    use_same_fen    = args.single_fen
    semihard_loss   = not args.softer_loss
    # # # # # #
    train_params = {
        'batch_size': batch_size,
        'opt_lr': opt_lr,
        'opt_betas': opt_betas,
        'opt_wd': opt_wd,
        'steplr_step': steplr_step,
        'steplr_gamma': steplr_gamma,
        'loss_margin': loss_margin,
        'semihard_loss': semihard_loss
    }

    # all trainable network parameters
    params = []

    # DF model config
    if resumed:
        model_config = resumed['config']
    else:
        with open(args.net_config, 'r') as fi:
            model_config = json.load(fi)
    
    # setup model config information
    model_name = model_config['model']
    if args.input_size is not None:
        model_config['input_size'] = args.input_size
        model_config['inflow_size'] = args.input_size
        model_config['outflow_size'] = args.input_size
    if args.features is not None:
        model_config['features'] = args.features
    features = model_config['features']
    feature_dim = model_config['feature_dim']
    model_arch = model_config['model']

    print("==> Model configuration:")
    print(json.dumps(model_config, indent=4))

    # initialize traffic feature extractor networks
    if model_arch.lower() == "espresso":
        inflow_fen = EspressoNet(input_channels=len(features),
                                **model_config)
    elif model_arch.lower() == "dcf":
        inflow_fen = Conv1DModel(input_channels=len(features),
                                input_size = model_config.get('inflow_size', 1000),
                                **model_config)
    elif model_arch.lower() == "laserbeak":
        inflow_fen = DFNet(input_channels=len(features),
                                input_size = model_config.get('inflow_size', 500),
                                **model_config)
    else:
        print(f"Invalid model architecture name \'{model_arch}\'!")
        sys.exit(-1)

    inflow_fen = inflow_fen.to(device)
    if resumed:
        inflow_fen.load_state_dict(resumed['inflow_fen'])
    params += inflow_fen.parameters()

    if use_same_fen:
        outflow_fen = inflow_fen

    else:
        if model_arch.lower() == "espresso":
            outflow_fen = EspressoNet(input_channels=len(features),
                                    **model_config)
        elif model_arch.lower() == "dcf":
            outflow_fen = Conv1DModel(input_channels=len(features),
                                    input_size = model_config.get('outflow_size', 1600),
                                    **model_config)
        elif model_arch.lower() == "laserbeak":
            outflow_fen = Conv1DModel(input_channels=len(features),
                                    input_size = model_config.get('outflow_size', 800),
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
    interval_size = model_config.get('interval_time', 0.03)
    processor = DataProcessor(features, interval_size = interval_size)

    with open(args.exp_config, 'r') as fi:
        data_config = json.load(fi)
    
    va_idx = np.arange(data_config['va_range'][0], 
                       data_config['va_range'][1])
    tr_idx = np.arange(data_config['tr_range'][0], 
                       data_config['tr_range'][1])

    # stream window definitions
    window_kwargs = model_config['window_kwargs']

    def make_dataloader(idx, shuffle=False):
        """
        """
        # load data from files
        if data_config['mode'] == 'pickle':
            samples = load_dataset_pkl(data_config['data_path'], 
                                       batch_list = idx)
        else:
            samples = load_dataset_text(data_config['data_path'], 
                                        sample_list = idx)

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
        checkpoint_fname = f'{model_name}_{time.strftime("%Y%m%d-%H%M%S")}'

    # create checkpoint directory if necesary
    if not os.path.exists(f'{checkpoint_dir}/{checkpoint_fname}/'):
        os.makedirs(f'{checkpoint_dir}/{checkpoint_fname}/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.online:
        all_criterion = OnlineCosineTripletLoss(margin = loss_margin, 
                                                semihard = semihard_loss)
        hard_criterion = OnlineHardCosineTripletLoss(margin = loss_margin, 
                                                     semihard = semihard_loss)
    else:
        criterion = CosineTripletLoss(margin = loss_margin, 
                                      semihard = semihard_loss)

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
                                "train_config": train_params,
                        }, checkpoint_path_epoch)

            history[epoch] = metrics

            if not args.online:  # generate new triplets
                gen_count = max(1, 33//window_kwargs['window_count']) \
                                if window_kwargs is not None else 33
                tr_data.generate_triplets(
                        fens = (inflow_fen, outflow_fen), 
                        margin = loss_margin if not args.hard else 0.,
                        gen_count = gen_count)

    except KeyboardInterrupt:
        pass

    finally:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # save epoch performance history
        results_fp = f'{log_dir}/{checkpoint_fname}.txt'
        with open(results_fp, 'w') as fi:
            json.dump(history, fi, indent='\t')
        
        # save a final checkpoint
        checkpoint_path_epoch = f"{checkpoint_dir}/{checkpoint_fname}/final.pth"
        print(f"Saving final checkpoint to {checkpoint_path_epoch}...")
        torch.save({
                        "epoch": epoch,
                        "inflow_fen": inflow_fen.state_dict(),
                        "outflow_fen": outflow_fen.state_dict(),
                        "opt": optimizer.state_dict(),
                        "config": model_config,
                        "train_config": train_params,
                }, checkpoint_path_epoch)
            
