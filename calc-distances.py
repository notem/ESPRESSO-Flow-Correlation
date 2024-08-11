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
from sklearn.metrics.pairwise import pairwise_distances

from utils.nets.espressonet import EspressoNet
from utils.layers import Mlp
from utils.data import BaseDataset, PairwiseDataset, load_dataset
from utils.processor import DataProcessor




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
                        #prog = 'WF Benchmark',
                        #description = 'Train & evaluate WF attack model.',
                        #epilog = 'Text at the bottom of help'
                        )

    # experiment configuration options
    parser.add_argument('--data_dir', 
                        default = '/data/path2', 
                        type = str,
                        help = "Path to dataset root 'pathx' directory.", 
                        required=True)
    parser.add_argument('--dists_file', 
                        default = '/data/path2', 
                        type = str,
                        help = "Path to dataset root 'pathx' directory.", 
                        required=True)
    parser.add_argument('--ckpt', 
                        default = None, 
                        type = str,
                        help = "Resume from checkpoint path.", 
                        )

    return parser.parse_args()



if __name__ == "__main__":
    """
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    dists_dir = os.path.basename(args.dists_file)
    if not os.path.exists(dists_dir):
        os.makedirs(dists_dir)

    # load checkpoint (if it exists)
    checkpoint_path = args.ckpt
    checkpoint_fname = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        resumed = torch.load(checkpoint_path)
        checkpoint_fname = os.path.basename(os.path.dirname(checkpoint_path))
    else:
        print("Failed to load model checkpoint!")
        sys.exit(-1)
    # else: checkpoint path and fname will be defined later if missing

    model_name = "DF"

    model_config = resumed['config']
    features = model_config['features']
    feature_dim = model_config['feature_dim']
    print(json.dumps(model_config, indent=4))

    # traffic feature extractor
    inflow_fen = EspressoNet(input_channels=len(features),
                            **model_config)
    inflow_fen = inflow_fen.to(device)
    inflow_fen.load_state_dict(resumed['inflow_fen'])
    inflow_fen.eval()

    if use_same_fen:
        outflow_fen = inflow_fen

    else:
        outflow_fen = EspressoNet(input_channels=len(features),
                                **model_config)
        outflow_fen = outflow_fen.to(device)
        outflow_fen.load_state_dict(resumed['outflow_fen'])
        outflow_fen.eval()

    # # # # # #
    # create data loaders
    # # # # # #
    # multi-channel feature processor
    processor = DataProcessor(features)

    te_idx = np.arange(0, 2000)
    va_idx = np.arange(2000, 4000)

    # stream window definitions
    window_kwargs = model_config['window_kwargs']

    def make_dataloader(idx, 
            sample_mode='undersample', 
            sample_ratio=None):
        """
        """
        # load data from files
        samples = load_dataset(args.data_dir, sample_list = idx)

        # build base dataset object
        data = BaseDataset(samples, processor,
                              window_kwargs = window_kwargs,
                          )
        #data = PairwiseDataset(data,
        #                       sample_mode = sample_mode,
        #                       sample_ratio = sample_ratio)
        return data

    print("Loading validation data...")
    va_data = make_dataloader(va_idx, 
                sample_mode = 'undersample', 
                sample_ratio = 1)
    va_ratio = len(va_data.uncorrelated_pairs) / len(va_data.correlated_pairs)
    print(f'Tr. data ratio: {va_ratio}')

    print("Loading testing data...")
    te_data = make_dataloader(te_idx,
                sample_ratio = None)
    te_ratio = len(te_data.uncorrelated_pairs) / len(te_data.correlated_pairs)
    print(f'Te. data ratio: {te_ratio}')

    def proc(t):
        """Pad and set tensor device
        """
        t = torch.nn.utils.rnn.pad_sequence(t, 
                                        batch_first=True, 
                                        padding_value=0.)
        return t.permute(0,2,1).float().to(device)


    def gen_embeds(data, in_fen, out_fen, proc):
        """
        """
        # Generate window embeddings for all samples
        inflow_embeds = []
        outflow_embeds = []
        for IDs in data.sample_ID_map.values():

            windows = data.windows[IDs['inflow']]
            embeds = in_fen(proc(windows)).detach().cpu()
            inflow_embeds.append(embeds)

            windows = data.windows[IDs['outflow']]
            embeds = out_fen(proc(windows)).detach().cpu()
            outflow_embeds.append(embeds)
        return inflow_embeds, outflow_embeds

    print("=> Generating embeds...")
    va_embeds, va_labels = gen_embeds(data, inflow_fen, outflow_fen, proc)
    te_embeds, te_labels = gen_embeds(data, inflow_fen, outflow_fen, proc)

    def build_sims(inflow_embeds, outflow_embeds):
        """
        """
        # Build window similarity & correlation matrices
        window_count = len(inflow_embeds[0])
        all_sims = np.zeros((len(inflow_embeds), len(outflow_embeds), len(window_count)))
        for i in range(len(inflow_embeds)):
            for j in range(len(outflow_embeds)):
                flow1 = inflow_embeds[i]
                flow2 = outflow_embeds[j]

                window_sims = F.cosine_similarity(flow1, flow2, dim=1)
                window_sims = window_sims.flatten().numpy(force=True)

                all_sims[i,j] = window_sims
        return all_sims

    print("=> Calculating sims...")
    va_corr, va_corr = build_sims(va_embeds, va_labels)
    te_sims, te_corr = build_sims(te_embeds, te_labels)

    with open(args.dists_file, 'wb') as fi:
        pickle.dump({'va_sims': va_sims, 
                    'va_corr': va_corr, 
                    'te_sims': te_sims, 
                    'te_corr': te_corr}, fi)

