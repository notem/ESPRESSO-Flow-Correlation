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
import pickle

from utils.nets.espressonet import EspressoNet
from utils.nets.dcfnet import Conv1DModel
from utils.layers import Mlp
from utils.data import BaseDataset, load_dataset
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
    use_same_fen = False

    dists_dir = os.path.dirname(args.dists_file)
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

    model_config = resumed['config']
    model_name = model_config['model']
    features = model_config['features']
    feature_dim = model_config['feature_dim']
    print(json.dumps(model_config, indent=4))

    # traffic feature extractor
    if model_name.lower() == "espresso":
        inflow_fen = EspressoNet(input_channels=len(features),
                                **model_config)
    elif model_name.lower() == 'dcf':
        inflow_fen = Conv1DModel(input_channels=len(features),
                                 input_size = 500,
                                 **model_config)
    inflow_fen = inflow_fen.to(device)
    inflow_fen.load_state_dict(resumed['inflow_fen'])
    inflow_fen.eval()

    if use_same_fen:
        outflow_fen = inflow_fen

    else:
        if model_config['model'].lower() == "espresso":
            outflow_fen = EspressoNet(input_channels=len(features),
                                    **model_config)
        elif model_config['model'].lower() == 'dcf':
            outflow_fen = Conv1DModel(input_channels=len(features),
                                    input_size = 800,
                                     **model_config)
        outflow_fen = outflow_fen.to(device)
        outflow_fen.load_state_dict(resumed['outflow_fen'])
        outflow_fen.eval()

    # # # # # #
    # create data loaders
    # # # # # #
    # multi-channel feature processor
    processor = DataProcessor(features)

    va_idx = np.arange(9000, 10000)
    te_idx = np.arange(10000, 15000)

    # stream window definitions
    window_kwargs = model_config['window_kwargs']

    def make_dataloader(idx):
        """
        """
        # load data from files
        samples = load_dataset(args.data_dir, sample_list = idx)

        # build base dataset object
        data = BaseDataset(samples, processor,
                              window_kwargs = window_kwargs,
                          )
        return data

    print("Loading validation data...")
    va_data = make_dataloader(va_idx)

    print("Loading testing data...")
    te_data = make_dataloader(te_idx)

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
        for ID in tqdm(data.IDs):#, description="Generating embeds..."):
            # inflow 
            windows = data.data[ID][0]
            embeds = in_fen(proc(windows)).squeeze().detach().cpu()
            inflow_embeds.append(embeds)
            # outflow
            windows = data.data[ID][1]
            embeds = out_fen(proc(windows)).squeeze().detach().cpu()
            outflow_embeds.append(embeds)
        return inflow_embeds, outflow_embeds

    print("=> Generating embeds...")
    va_inflow_embeds, va_outflow_embeds = gen_embeds(va_data, inflow_fen, outflow_fen, proc)
    te_inflow_embeds, te_outflow_embeds = gen_embeds(te_data, inflow_fen, outflow_fen, proc)

    def build_sims(inflow_embeds, outflow_embeds):
        """
        """
        # Build window similarity & correlation matrices
        window_count = len(inflow_embeds[0])
        all_sims = np.zeros((len(inflow_embeds), len(outflow_embeds), window_count))
        with tqdm(total=len(inflow_embeds)*len(outflow_embeds)) as pbar:
            pbar.set_description("Building similarities matrix...")
            for i in range(len(inflow_embeds)):
                for j in range(len(outflow_embeds)):
                    flow1 = inflow_embeds[i]
                    flow2 = outflow_embeds[j]

                    window_sims = F.cosine_similarity(flow1, flow2, dim=1)
                    window_sims = window_sims.flatten().numpy(force=True)

                    all_sims[i,j] = window_sims
                    pbar.update(1)
        return all_sims

    print("=> Calculating sims...")
    va_sims = build_sims(va_inflow_embeds, va_outflow_embeds)
    te_sims = build_sims(te_inflow_embeds, te_outflow_embeds)

    with open(args.dists_file, 'wb') as fi:
        pickle.dump({'va_sims': va_sims, 
                    'te_sims': te_sims}, fi)

