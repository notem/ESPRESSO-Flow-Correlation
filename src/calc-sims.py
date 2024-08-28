import torch
import numpy as np
import os
from tqdm import tqdm
import json
import argparse
import pickle

from utils.nets.espressonet import EspressoNet
from utils.nets.dcfnet import Conv1DModel
from utils.nets.transdfnet import DFNet
from utils.data import *
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
                        prog = 'calc-sims.py',
                        description = 'Generate the similarity matrix for the validation \
                                        & testing datasets using a trained FEN.',
                        #epilog = 'Text at the bottom of help'
                        )

    # experiment configuration options
    parser.add_argument('--exp_config',
                        default = './configs/exps/june.json',
                        type = str,
                        help = "Path to JSON config file containing dataset configuration.", 
                        required=True
                    )
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
        import sys
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
                                input_size = model_config.get('inflow_size', 1000),
                                **model_config)
    elif model_name.lower() == 'laserbeak':
        inflow_fen = DFNet(input_channels=len(features),
                                input_size = model_config.get('inflow_size', 500),
                                **model_config)
    inflow_fen = inflow_fen.to(device)
    inflow_fen.load_state_dict(resumed['inflow_fen'])
    inflow_fen.eval()

    if use_same_fen:
        outflow_fen = inflow_fen

    else:
        if model_name.lower() == "espresso":
            outflow_fen = EspressoNet(input_channels=len(features),
                                    **model_config)
        elif model_name.lower() == "dcf":
            outflow_fen = Conv1DModel(input_channels=len(features),
                                    input_size = model_config.get('outflow_size', 1600),
                                    **model_config)
        elif model_name.lower() == "laserbeak":
            outflow_fen = Conv1DModel(input_channels=len(features),
                                    input_size = model_config.get('outflow_size', 800),
                                    **model_config)
        outflow_fen = outflow_fen.to(device)
        outflow_fen.load_state_dict(resumed['outflow_fen'])
        outflow_fen.eval()

    # # # # # #
    # create data loaders
    # # # # # #
    # multi-channel feature processor
    processor = DataProcessor(features)

    with open(args.exp_config, 'r') as fi:
        data_config = json.load(fi)
    
    te_idx = np.arange(data_config['te_range'][0], 
                       data_config['te_range'][1])
    va_idx = np.arange(data_config['va_range'][0], 
                       data_config['va_range'][1])

    # stream window definitions
    window_kwargs = model_config['window_kwargs']

    def make_dataloader(idx):
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
        for ID in tqdm(data.IDs, desc="Generating embeddings..."):
            # inflow 
            windows = data.data[ID][0]
            embeds = in_fen(proc(windows)).squeeze().detach().cpu()
            inflow_embeds.append(embeds)
            # outflow
            windows = data.data[ID][-1]
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
        inflow_embeds = torch.stack(inflow_embeds, dim=0)
        outflow_embeds = torch.stack(outflow_embeds, dim=0)

        # normalize embedding dimension by vector norm
        inflow_embeds = inflow_embeds / torch.norm(inflow_embeds, p=2, dim=-1, keepdim=True)
        outflow_embeds = outflow_embeds / torch.norm(outflow_embeds, p=2, dim=-1, keepdim=True)

        # smash together matrices as dot-product to produce the cosine similarity of sampleXwindows
        all_sims = torch.matmul(inflow_embeds.permute(1,0,2),
                               outflow_embeds.permute(1,2,0))
        all_sims = all_sims.permute(1,2,0).numpy(force=True)
        
        return all_sims

    print("=> Calculating sims...")
    va_sims = build_sims(va_inflow_embeds, va_outflow_embeds)
    te_sims = build_sims(te_inflow_embeds, te_outflow_embeds)

    with open(args.dists_file, 'wb') as fi:
        pickle.dump({'va_sims': va_sims.astype(np.float16), 
                    'te_sims': te_sims.astype(np.float16)}, fi)

