from functools import partial
import numpy as np
import math
from tqdm import tqdm
import pickle


def split_round_robin(sequence, 
                      split_count = 2):
    """Vectorized (fast) implementation of round-robin style traffic splitting.

    Args:
        sequence (np.array): Flow sequence as np.array of size (N,3)
        split_count (int, optional): Number of circuits to split traffic across. Defaults to 2.

    Returns:
        list of np.array: List of split flows
    """
    split_idx = np.tile(np.arange(split_count), 
                        math.ceil(len(sequence)/split_count))[:len(sequence)]
    return [sequence[split_idx == i] for i in range(split_count)]
    
    
def split_random(sequence, 
                 split_count = 2):
    """Vectorized (fast) implementation of random (per-packet) traffic splitting.

    Args:
        sequence (np.array): Flow sequence as np.array of size (N,3)
        split_count (int, optional): Number of circuits to split traffic across. Defaults to 2.

    Returns:
        list of np.array: List of split flows
    """
    split_idx = np.random.randint(split_count, size=len(sequence))
    return [sequence[split_idx == i] for i in range(split_count)]


def split_weighted_random(sequence, 
                          split_count = 2):
    """Vectorized (fast) implementation of weighted random traffic splitting.

    Args:
        sequence (np.array): Flow sequence as np.array of size (N,3)
        split_count (int, optional): Number of circuits to split traffic across. Defaults to 2.

    Returns:
        list of np.array: List of split flows
    """
    alpha = np.ones(split_count)*split_count
    split_idx = np.argmax(np.random.dirichlet(alpha, 
                                              size = len(sequence)),
                          axis=-1)
    return [sequence[split_idx == i] for i in range(split_count)]


def split_batched_weighted_random(sequence, 
                                  split_count = 2, 
                                  batch_range = (50,71)):
    """Vectorized (fast) implementation of batched weighted random traffic splitting.

    Args:
        sequence (np.array): Flow sequence as np.array of size (N,3)
        split_count (int, optional): Number of circuits to split traffic across. Defaults to 2.
        batch_range (tuple, optional): Range of batch sizes to uniformly sample. Defaults to (50,71).

    Returns:
        list of np.array: List of split flows
    """
    # generate array of batch sizes
    max_size = math.ceil(len(sequence)/batch_range[0])
    bs = np.random.randint(*batch_range, size = max_size)
    # generate array of split idx for each batch
    alpha = np.ones(split_count)*split_count
    bs_idx = np.argmax(np.random.dirichlet(alpha, 
                                           size = len(bs)), 
                       axis=-1)
    
    # build the split indices via iteration (maybe room to optimize further?)
    split_idx = []
    seq_len = len(sequence)
    for i,b in enumerate(bs):
        split_idx.append(np.repeat(bs_idx[i], 
                                   repeats = min(b, seq_len)))
        seq_len -= b
        if seq_len <= 0:
            break
    split_idx = np.concatenate(split_idx)
    
    return [sequence[split_idx == i] for i in range(split_count)]


SPLIT_FUNCS = {'round-robin': split_round_robin, 
               'random': split_random,
               'weighted-random': split_weighted_random,
               'batched-weighted-random': split_batched_weighted_random}


def proc_dataset(data, split_func):
    """
    Apply the TrafficSliver defense to inflow samples for all sample pairs in a dataset file.
    """
    split_data = dict()
    
    total = sum(len(values) for values in data.values())
    with tqdm(total=total) as pbar:
        
        for batch_idx in data.keys():
            samples = data[batch_idx]
            new_samples = []
            
            pbar.set_description(f"Processing batch {batch_idx}...")
            for i,(inflow, outflow) in enumerate(samples):
                inflow_split = split_func(inflow)
                
                correlated_flows = []
                correlated_flows.extend(inflow_split)   # first n-1 flows are inflows
                correlated_flows.append(outflow)        # last flow is outflow
                
                new_samples.append(correlated_flows)
                pbar.update(1)
                
            split_data[batch_idx] = new_samples
            
    return split_data


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(
                        prog = 'ts-sim.py',
                        description = 'Apply the simulated TrafficSliver defense on a packaged flow correlation dataset.',
    )
    parser.add_argument('--infile', type=str)
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--strategy', choices=["round-robin", 
                                               "random", 
                                               "weighted-random", 
                                               "batched-weighted-random"],
                        default="batched-weighted-random")
    parser.add_argument('--splits', type=int, default=2)
    args = parser.parse_args()
    
    # load pickle file containing flow dataset
    with open(args.infile, 'rb') as fi:
        data = pickle.load(fi)
    
    # apply split strategy to all inflows in the dataset
    data = proc_dataset(data, partial(SPLIT_FUNCS[args.strategy], 
                                      split_count=args.splits))
    
    # save out the TS-variant dataset
    with open(args.outfile, 'wb') as fi:
        pickle.dump(data, fi)