import numpy as np
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import pickle
import random

from utils.processor import DataProcessor


class BaseDataset(data.Dataset):
    """
    Dataset object to process and hold windowed samples in a convenient package.

    Attributes
    ----------
    data_ID_tuples : list
        List of sample circuit-level identifiers as unique tuples, (cls_no, sample_no, circuit_no)
    data_windows : dict
        Dictionary containing lists of stream windows for streams within each chain
    """
    def __init__(self, samples, data_processor,
                       window_kwargs = None,
                       preproc_feats = False,
                       ):
        """
        Load the metadata for samples collected in our SSID data. 

        Parameters
        ----------
        data_dict : dict
            Dictionary of datasamples produced by load_dataset()
        data_processor : DataProcessor
            The processor object that convert raw samples into their feature representations.
        window_kwargs : dict
            Dictionary containing the keyword arguments for the window processing function
        preproc_feats : bool
            If True, the data processor will be applied on samples before windowing.
        """

        # useful info to have on hand...
        self.window_count = window_kwargs['window_count'] if window_kwargs is not None else 1

        self.IDs = []       # full list of unique identifiers for circuit streams
        self.data = dict()  # link ID to stream content

        times_processor = DataProcessor(('times',))

        # loop through the samples within each class
        for ID,sample_tuple in tqdm(enumerate(samples), 
                                    total=len(samples), 
                                    dynamic_ncols=True, 
                                    desc="Preparing dataset..."):

            self.IDs.append(ID)
            self.data[ID] = []

            # loop through split circuits within each sample
            for sample in sample_tuple:

                if preproc_feats:  # apply processing before windowing if enabled
                    sample = data_processor(sample)

                # split the circuit data into windows if window_kwargs have been provided
                if window_kwargs is not None:
                    if len(sample) > 0:
                        times = times_processor(sample)   # pkt times
                        windows = create_windows(times, sample, **window_kwargs)  # windowed stream

                        if not preproc_feats:  # apply processing if not yet performed
                            # create multi-channel feature representation of windows independently
                            for i in range(len(windows)):
                                windows[i] = data_processor(windows[i])
                        self.data[ID].append(windows)
                    else:
                        self.data[ID].append([torch.empty((0,data_processor.input_channels)) 
                                              for _ in range(len(windows))])

                else:  # no windowing
                    if not preproc_feats:  # apply processing if not yet performed
                        sample = data_processor(sample)
                    self.data[ID].append([sample])
        self.IDs = np.array(self.IDs)

    def __len__(self):
        """
        Count of all streams within the dataset.
        """
        return len(self.IDs)

    def __getitem__(self, index):
        """
        """
        ID = self.IDs[index]
        samples = self.data[ID]
        return samples, ID


class TripletDataset(BaseDataset):
    """
    Dataset object for generating random triplets for triplet learning.
    """
    def __init__(self, dataset):
        """
        Initialize triplet dataset from an existing SSI dataset object.
        """
        vars(self).update(vars(dataset))  # load dataset attributes

        # create and shuffle indices for samples
        self.generate_triplets()

    def __len__(self):
        """
        An epoch of the TripletDataset iterates over all samples within partition_1
        """
        return len(self.triplets)

    def __getitem__(self, index):
        """
        Get a Triplet sample with a randomly selected window.

        Parameters
        ----------
        index : int

        Returns
        -------
        anc : tensor
            Window to represent the anchor sample
        pos : tensor
            Correlated window to represent positive sample
        neg : tensor
            Uncorrelated window to represent the negative sample
        """
        anc_ID, pos_ID, neg_ID = self.triplets[index][0], self.triplets[index][1], self.triplets[index][2]

        # get windows for anc, pos, neg
        anc = self.data[anc_ID[0]][0]
        pos = self.data[pos_ID[0]][-1]
        neg = self.data[neg_ID[0]][-1]

        # randomly select a window for the triplet (skipping empty windows in anchor)
        if anc_ID[1] < 0:
            candidate_idx = [i for i,window in enumerate(anc) if len(window) > 0]
            if len(candidate_idx) <= 0:
                candidate_idx = list(range(len(anc)))
            anc_window_idx = np.random.choice(candidate_idx)
        else:
            anc_window_idx = anc_ID[1]

        if neg_ID[1] < 0:
            candidate_idx = [i for i,window in enumerate(neg) if len(window) > 0]
            if len(candidate_idx) <= 0:
                candidate_idx = list(range(len(neg)))
            neg_window_idx = np.random.choice(candidate_idx)
        else:
            neg_window_idx = neg_ID[1]

        return anc[anc_window_idx], pos[anc_window_idx], neg[neg_window_idx]

    def _trim_triplets(self, count):
        random.shuffle(self.triplets)
        if len(self.triplets) > count:
            self.triplets = self.triplets[:count]

    def generate_triplets(self, 
                          fens = None, 
                          margin = 0, 
                          max_triplets = None, 
                          gen_count = 1,
                          on_cpu = False):
        """
        """
        self.triplets = []

        # shuffle sample IDs
        all_idx = np.arange(len(self.IDs))
        np.random.shuffle(all_idx)
        all_IDs = self.IDs[all_idx]
        
        # first-half of shuffled IDs will be used for anc,pos pairs; second-half as neg samples
        split_point = len(all_idx)//2
        pos_IDs = all_IDs[:split_point]
        neg_IDs = all_IDs[split_point:]

        # offline triplet (semi)hard mining
        if fens is not None:
            # use fen to create embeddings for all samples in the dataset
            iter_total = len(all_IDs)*2 + split_point*self.window_count
            with tqdm(total = iter_total, dynamic_ncols = True) as pbar:
                pbar.set_description('Generating embeddings...')

                window_sample_idx = None
                inflow_embeds = []
                outflow_embeds = []
                valid_windows = []    # use for filtering out empty windows during neg selection
                for ID in all_IDs:
                    samples = self.data[ID]
                    if len(samples) > 2:
                        outflow_sample = samples[-1]
                        inflow_sample = samples[np.random.randint(len(samples)-1)]
                        samples = [inflow_sample, outflow_sample]
                    for i,windows in enumerate(samples):
                        if i == len(samples)-1:   # outflow sample
                            fen = fens[1]
                        else:            # inflow sample
                            fen = fens[0]
                        windows = pad_sequence(windows, 
                                              batch_first = True, 
                                              padding_value = 0.)
                        windows = windows.permute(0,2,1).to(fen.get_device())
                        # hopefully your memory is adequate to fit all windows into memory...
                        window_embeds = fen(windows).detach()
                        if on_cpu:
                            # do work on CPU, which probably has more access to memory?
                            window_embeds = window_embeds.cpu()
                        
                        # if using ESPRESSO style FEN, sample a limited number of 
                        # windows to prevent memory explosion
                        if len(window_embeds.size()) == 3:
                            if window_sample_idx is None:
                                # generate vector to randomly select 16 windows from all window idx
                                window_sample_idx = np.arange(window_embeds.size(1))
                                np.random.shuffle(window_sample_idx)
                                window_sample_idx = window_sample_idx[:16]
                            # slice out selected windows
                            window_embeds = window_embeds[:,window_sample_idx,:]
                                
                        if i == len(samples)-1:
                            outflow_embeds.append(window_embeds)
                            # never sample an empty window as the negative
                            # (guarantees avoidance of both pos and neg windows being empty)
                            valid_windows.append([len(w) > 0 for w in windows])
                        else:
                            inflow_embeds.append(window_embeds)
                        pbar.update(1)

                pbar.set_description(f'Calculating similarity matrix...')
                ratio = int(len(inflow_embeds) / len(outflow_embeds))
                inflow_embeds = torch.stack(inflow_embeds, dim=0)
                outflow_embeds = torch.stack(outflow_embeds, dim=0)
                valid_windows = torch.Tensor(valid_windows)[split_point:].bool()

                # normalize embedding dim
                inflow_embeds = inflow_embeds / torch.norm(inflow_embeds, p=2, 
                                                           dim=-1, keepdim=True)
                outflow_embeds = outflow_embeds / torch.norm(outflow_embeds, p=2, 
                                                             dim=-1, keepdim=True)

                # calculate pairwise similarities
                if len(inflow_embeds.size()) == 3:
                    # permute axis and dot-product across batch and embed dim
                    all_sim = torch.matmul(inflow_embeds.permute(1,0,2),
                                           outflow_embeds.permute(1,2,0))
                    all_sim = all_sim.permute(1,2,0) # (N,sim,windows)

                elif len(inflow_embeds.size()) == 4:
                    # espresso-style networks will have extra window dimension to handle
                    all_sim = torch.matmul(inflow_embeds.permute(1,2,0,3),
                                           outflow_embeds.permute(1,2,3,0))
                    all_sim = all_sim.permute(2,3,0,1)  # (N,sim,1,windows)
                    
                all_sim = all_sim.cpu()

                mode = 'hard' if margin <= 0 else 'semi-hard'
                pbar.set_description(f'Finding {mode} triplets...')
                # identify (semi)hard triplets
                # select one random negative for every (anc,pos) pair
                for i,pos_ID in enumerate(pos_IDs):
                    for window_idx in range(self.window_count):
                        pos_anc_tuple = (pos_ID, window_idx)

                        # current window positive sim
                        pos_sim = all_sim[i,i//ratio,window_idx]
                        if len(inflow_embeds.size()) == 4:    
                            # if using espresso-style network, extra dim needs to be reduced
                            #pos_sim = torch.amin(pos_sim, dim=-1)   # hardest positive sim
                            pos_sim = torch.mean(pos_sim, dim=-1)   # avg. pos sim
                            
                        # sims for all neg. candidates
                        neg_sims = all_sim[i,split_point:]
                        if len(inflow_embeds.size()) == 4:
                            #neg_sims = torch.amax(neg_sims, dim=-1)    # hardest neg. sim
                            neg_sims = torch.mean(neg_sims, dim=-1)    # avg. neg sim
                            
                        # include hard & semi-hard triplets (not true semi-hard mining)
                        valid_neg_idx = torch.where((neg_sims + margin > pos_sim) & valid_windows)
                        
                        # build triplet combinations
                        candidate_count = len(valid_neg_idx[0])
                        if candidate_count > 0:
                            js = np.random.choice(np.arange(candidate_count), 
                                             size = min(gen_count, candidate_count), 
                                             replace = False)
                            for j in js:
                                # build negative identifier tuple
                                neg_idx = valid_neg_idx[0].numpy(force=True)[j]
                                neg_window_idx = valid_neg_idx[1].numpy(force=True)[j]
                                neg_tuple = (neg_IDs[neg_idx], neg_window_idx)
                                # add triplet
                                self.triplets.append((pos_anc_tuple, pos_anc_tuple, neg_tuple))
                        pbar.update(1)

        if len(self.triplets) > 0:
            if max_triplets is not None:
                self._trim_triplets(max_triplets)
            return
        # no candidate triplets? randomly sample triplets

        # mine random triplets
        # build numpy vectors 
        a = np.repeat(self.IDs, self.window_count)
        b = np.tile(np.arange(self.window_count), len(self.IDs))
        c = np.copy(a)  # selected neg_idx
        np.random.shuffle(c)
        for i in range(len(c)):
            if c[i] == a[i]:
                c[i] = c[i] + 1 % len(self.IDs)
        anc_pos = np.stack((a, b), axis=-1)
        neg = np.stack((c, np.ones_like(c)*-1), axis=-1)
        self.triplets = np.stack((anc_pos, anc_pos, neg), axis=1)

        if max_triplets is not None:
            self._trim_triplets(max_triplets)

    @staticmethod
    def batchify(batch):
        """
        convert samples to tensors and pad samples to equal length
        """
        # re-arrange into proper batches
        batch_anc = []
        batch_neg = []
        batch_pos = []
        for i in range(len(batch)):
            anc, pos, neg = batch[i]
            batch_anc.append(anc)
            batch_neg.append(neg)
            batch_pos.append(pos)
        batch_x = (batch_anc, batch_pos, batch_neg)
    
        # pad batches and fix dimension
        batch_x_tensors = []
        for batch_x_n in batch_x:
            batch_x_n = nn.utils.rnn.pad_sequence(batch_x_n, 
                                                  batch_first = True, 
                                                  padding_value = 0.)
            if len(batch_x_n.shape) < 3:  # add channel dimension if missing
                batch_x_n = batch_x_n.unsqueeze(-1)
            batch_x_n = batch_x_n.permute(0,2,1)
            batch_x_n = batch_x_n.float()
            batch_x_tensors.append(batch_x_n)
    
        return batch_x_tensors


class OnlineDataset(BaseDataset):
    """
    """
    def __init__(self, dataset):
        """
        Initialize triplet dataset from an existing SSI dataset object.
        """
        vars(self).update(vars(dataset))  # load dataset attributes

        # create and shuffle indices for samples
        np.random.shuffle(self.IDs)

    def __getitem__(self, index):
        """
        """
        ID = self.IDs[index]
        samples = self.data[ID]
        if len(samples) == 2:
            return samples
        else:
            outflow_sample = samples[-1]
            inflow_sample = samples[np.random.randint(len(samples)-1)]
            return [inflow_sample, outflow_sample]

    @staticmethod
    def batchify(batch):
        """
        convert samples to tensors and pad samples to equal length
        """
        batch_x_inflow = []
        batch_x_outflow = []
        for i in range(len(batch)):
            batch_x_inflow.append(batch[i][0])
            batch_x_outflow.append(batch[i][-1])

        # pick a random window to return
        window_count = len(batch_x_inflow[0])
        window_idx = np.random.choice(range(window_count))
        batch_x_inflow = [x[window_idx] for x in batch_x_inflow]
        batch_x_outflow = [x[window_idx] for x in batch_x_outflow]

        # pad batches and fix dimension
        batch_x_inflow_tensor = pad_sequence(batch_x_inflow, 
                                                batch_first = True, 
                                                padding_value = 0.)
        batch_x_inflow_tensor = batch_x_inflow_tensor.permute(0,2,1).float()
        batch_x_outflow_tensor = pad_sequence(batch_x_outflow, 
                                                batch_first = True, 
                                                padding_value = 0.)
        batch_x_outflow_tensor = batch_x_outflow_tensor.permute(0,2,1).float()
    
        return batch_x_inflow_tensor, batch_x_outflow_tensor



def create_windows(times, features,
                    window_width = 5,
                    window_count = 11,
                    window_overlap = 2,
                    include_all_window = False,
                ):
    """
    Slice a sample's full stream into time-based windows.

    Parameters
    ----------
    times : ndarray
    features : ndarray
    window_width : int
    window_count : int
    window_overlap : int

    Returns
    -------
    list
        A list of stream windows as torch tensors.
    """
    window_features = []

    if include_all_window:
        window_count -= 1

    if window_count > 0:
        window_step = min(window_width - window_overlap, 1)

        # Create overlapping windows
        for start in np.arange(0, stop = window_count * window_step, 
                                  step = window_step):

            end = start + window_width

            window_idx = torch.where(torch.logical_and(times >= start, times < end))[0]
            window_features.append(features[window_idx])

    # add full stream as window
    if include_all_window:
        window_features.append(features)

    return window_features


def load_sample_text(pth):
    """Read in a plaintext trace file
    """
    sample = []
    with open(pth, 'r') as fi:
        for line in fi:
            ts, sizedir = line.strip().split('\t')

            # filter out probable ack-only packets
            if abs(int(sizedir)) < 100:
                continue

            # merge packets with same direction at same timestamp
            if len(sample) > 0 and \
                    sample[-1][0] == float(ts) and \
                    sample[-1][2] == np.sign(int(sizedir)):
                sample[-1][1] += abs(float(sizedir))
            else:
                sample.append([float(ts),                   # time
                                abs(float(sizedir)),        # size
                                np.sign(int(sizedir))])     # dir

    return torch.tensor(sample)


def load_dataset_text(root_dir,
        sample_list = np.arange(10000),
        seed = 0):
    """
    Load traffic sliver dataset from files.

    Parameters
    ----------
    root_dir : str

    Returns
    -------
    dict
        A class-keyed dictionary containing all circuit samples within a nested list
    """
    samples = []

    inflow_dir = os.path.join(root_dir, 'inflow')
    outflow_dir = os.path.join(root_dir, 'outflow')

    # template of filenames for circuit stream data
    inflow_files = os.listdir(inflow_dir)
    random.seed(seed)
    random.shuffle(inflow_files)

    # filter based on selected idx
    inflow_files = np.array(inflow_files, dtype=object)[sample_list].tolist()

    for filename in inflow_files:
        inflow_sample_pth = os.path.join(inflow_dir, filename)
        outflow_sample_pth = os.path.join(outflow_dir, filename)

        if os.path.exists(outflow_sample_pth):

            inflow_trace = load_sample_text(inflow_sample_pth)
            outflow_trace = load_sample_text(outflow_sample_pth)

            samples.append([inflow_trace, outflow_trace])

    return samples


def load_dataset_pkl(pkl_file, 
        batch_list = np.arange(100),
        seed = 0):
    """Load data from a pickle file by batches
    """
    with open(pkl_file, 'rb') as fi:
        data = pickle.load(fi)
        
    all_batches = list(data.keys())
    random.seed(seed)
    random.shuffle(all_batches)
    
    samples = []
    for i in batch_list:
        for flow_pair in data[all_batches[i]]:
            samples.append([torch.tensor(s) for s in flow_pair])
        
    return samples


if __name__ == "__main__":

    root = '../data/undefended'

    # chain-based sample splitting
    tr_idx = np.arange(0,10000)
    te_idx = np.arange(10000,15000)

    # stream window definitions
    #window_kwargs = {
    #                 'window_width': 5, 
    #                 'window_count': 11, 
    #                 'window_overlap': 2
    #                 }
    window_kwargs = None   # disable windowing (e.g., create one window with all features)

    # multi-channel feature processor
    processor = DataProcessor(('sizes', 'iats', 'time_dirs', 'dirs'))

    # load dataset object
    for idx in (te_idx, tr_idx):

        # load data from files
        samples = load_dataset_text(root, sample_list = idx)

        # build base dataset object
        data = BaseDataset(samples, processor,
                              window_kwargs = window_kwargs,
                          )
        print(f'Streams: {len(data)}')
        for windows, sample_ID in tqdm(data, desc="Looping over circuits..."):
            pass

        # construct a triplets dataset object, derived from the base dataset object
        # Note: changes to the base dataset propogate to the triplet object once initialized (I think?)
        triplets = TripletDataset(data)
        print(f'Triplets: {len(triplets)}')
        for anc, pos, neg in tqdm(triplets, desc="Looping over triplets..."):
            pass
        triplets.generate_triplets()  # resample partition split after each full loop over dataset!
