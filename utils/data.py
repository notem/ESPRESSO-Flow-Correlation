import numpy as np
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import pickle
import itertools
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
                    times = times_processor(sample)   # pkt times
                    windows = create_windows(times, sample, **window_kwargs)  # windowed stream

                    if not preproc_feats:  # apply processing if not yet performed
                        # create multi-channel feature representation of windows independently
                        for i in range(len(windows)):
                            if len(windows[i]) <= 0:
                                windows[i] = torch.empty((0,data_processor.input_channels))
                            else:
                                windows[i] = data_processor(windows[i])
                    self.data[ID].append(windows)

                else:  # no windowing
                    if not preproc_feats:  # apply processing if not yet performed
                        sample = data_processor(sample)
                    self.data[ID].append([sample])


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
        anc_ID, pos_ID, neg_ID = self.triplets[index]

        # get windows for anc, pos, neg
        anc = self.data[anc_ID[0]][0]
        pos = self.data[pos_ID[0]][1]
        neg = self.data[neg_ID[0]][1]

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

    def generate_triplets(self, fens=None, margin=0, max_triplets=None):
        """
        """
        self.triplets = []

        # offline triplet (semi)hard mining
        if fens is not None:
            # use fen to create embeddings for all samples in the dataset
            with tqdm(total=len(self.IDs)*2+len(self.IDs),
                      dynamic_ncols=True) as pbar:

                pbar.set_description('Generating embeddings...')
                inflow_embeds = []
                outflow_embeds = []
                for ID in self.IDs:
                    samples = self.data[ID]
                    for i,sample in enumerate(samples):
                        if i % 2 == 0:   # inflow sample
                            fen = fens[0]
                        else:            # outflow sample
                            fen = fens[1]
                        sample = pad_sequence(sample, 
                                              batch_first = True, 
                                              padding_value = 0.)
                        sample = sample.permute(0,2,1).to(fen.get_device())
                        # hopefully your GPU memory is adequate to fit all windows into memory...
                        window_embeds = fen(sample).detach().cpu()
                        if i % 2 == 0:
                            inflow_embeds.append(window_embeds)
                        else:
                            outflow_embeds.append(window_embeds)
                        pbar.update(1)

                pbar.set_description(f'Calculating similarity matrix...')
                inflow_embeds = torch.stack(inflow_embeds, dim=0)
                outflow_embeds = torch.stack(outflow_embeds, dim=0)

                # normalize embedding dim
                inflow_embeds = inflow_embeds / torch.norm(inflow_embeds, p=2, dim=-1, keepdim=True)
                outflow_embeds = outflow_embeds / torch.norm(outflow_embeds, p=2, dim=-1, keepdim=True)

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

                mode = 'hard' if margin <= 0 else 'semi-hard'
                pbar.set_description(f'Finding {mode} triplets...')
                # identify (semi)hard triplets
                # select one random negative for every (anc,pos) pair
                for i,ID in enumerate(self.IDs):
                    for window_idx in range(all_sim.size(2)):
                        pos_anc_tuple = (ID, window_idx)

                        # current window positive sim
                        pos_sim = all_sim[i,i,window_idx]
                        if len(inflow_embeds.size()) == 4:    
                            # if using espresso-style network, extra dim needs to be reduced
                            pos_sim = np.amin(pos_sim)  # hardest positive sim
                            
                        neg_sims = all_sim[i]
                        if len(inflow_embeds.size()) == 4:
                            neg_sims = np.amax(neg_sims, dim=-1)
                            
                        if margin > 0: # semi-hard mining
                            valid_neg_idx = torch.where((neg_sims + margin > pos_sim) & 
                                                        (neg_sims < pos_sim))
                        else:  # hard mining
                            valid_neg_idx = torch.where(neg_sims > pos_sim)
                            
                        if len(valid_neg_idx[0]) > 0:
                            # select random idx
                            j = np.random.randint(0, len(valid_neg_idx[0]))
                            
                            # build negative identifier tuple
                            neg_tuple = (valid_neg_idx[0].numpy(force=True)[j], 
                                         valid_neg_idx[1].numpy(force=True)[j])
                            
                            # add triplet
                            self.triplets.append((pos_anc_tuple, pos_anc_tuple, neg_tuple))
                    pbar.update(1)

        if len(self.triplets) > 0:
            if max_triplets is not None:
                self._trim_triplets(max_triplets)
            return
        # no candidate triplets? randomly sample triplets

        # setup random window triplets with no mining
        with tqdm(total=len(self.IDs)*self.window_count, 
                  dynamic_ncols=True) as pbar:
            pbar.set_description(f'Generating random triplets...')
            for index in range(len(self.IDs)):
                ID = self.IDs[index]    # chain ID for anchor & positive
                for i in range(self.window_count):
                    neg_ID = None  # pick a random negative ID
                    while neg_ID != ID:
                        neg_ID = random.choice(self.IDs)
                    self.triplets.append(((ID, i), (ID, i), (neg_ID, -1)))
                    pbar.update(1)
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
        return self.data[ID]

    @staticmethod
    def batchify(batch):
        """
        convert samples to tensors and pad samples to equal length
        """
        batch_x_inflow = []
        batch_x_outflow = []
        for i in range(len(batch)):
            batch_x_inflow.append(batch[i][0])
            batch_x_outflow.append(batch[i][1])

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


def load_sample(pth):
    """Read in a plaintext trace file
    """
    sample = []
    with open(pth, 'r') as fi:
        for line in fi:
            ts, sizedir = line.strip().split('\t')
            sample.append((float(ts), abs(float(sizedir)), np.sign(int(sizedir))))
    #return np.array(sample)
    return torch.tensor(sample)


def load_dataset(root_dir,
        sample_list = np.arange(10000)):
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

    # filter based on selected idx
    inflow_files = np.array(inflow_files, dtype=object)[sample_list].tolist()

    for filename in inflow_files:
        inflow_sample_pth = os.path.join(inflow_dir, filename)
        outflow_sample_pth = os.path.join(outflow_dir, filename)

        if os.path.exists(outflow_sample_pth):

            inflow_trace = load_sample(inflow_sample_pth)
            outflow_trace = load_sample(outflow_sample_pth)

            samples.append([inflow_trace, outflow_trace])

    return samples


if __name__ == "__main__":

    root = '../data/undefended'

    # chain-based sample splitting
    te_idx = np.arange(0,1000)
    tr_idx = np.arange(1000,11000)

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
        samples = load_dataset(root, sample_list = idx)

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
