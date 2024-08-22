import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from tqdm import tqdm

import argparse
import os
import pickle

# strategically selected Ks for smooth curve
DEFAULT_K = (1,2,4,8,16,32,64,
             128,256,512,
             1024,2048,4096)
DEFAULT_VOTE_THRS = (0.7, 0.8, 0.9)


def calc_votes_thr(sims, sorted_idx, k=1):
    """
    """
    local_thr_idx = sorted_idx[:,k,:]
    local_thr = np.take_along_axis(sims, local_thr_idx[:,np.newaxis,:], 
                                   axis=1).squeeze()
    return (sims >= local_thr).astype(bool)


def tally_votes(votes, perc=0.8):
    """
    """
    window_count = votes.shape[-1]
    required_votes = math.ceil(perc * window_count)

    votes = np.sum(votes, axis=-1)
    corr_pred = (votes >= required_votes)

    return corr_pred


def calc_metrics(corr_true, corr_pred):
    """
    """
    corr_true = corr_true.astype(bool)
    corr_pred = corr_pred.astype(bool)
    TP = (corr_pred & corr_true).sum()
    TN = (~corr_pred & ~corr_true).sum()
    FP = (corr_pred & ~corr_true).sum()
    FN = (~corr_pred & corr_true).sum()
    return TP, TN, FP, FN


def calc_min_sim(sims, corr, perc=1.0):
    """Calculate the minimum acceptable similarity threshold 
        using the percentile of known correlated windows
    """
    return np.percentile(sims[corr], int(perc*100))


def evaluate(sims, 
        ks = DEFAULT_K, 
        min_sims = (None,),
        vote_thrs = DEFAULT_VOTE_THRS,
        keep_intermediate = False,
        ):
    """
    """
    fpr, tpr, cm = [], [], []
    
    # correlated when i=j
    corr_true = np.eye(sims.shape[0], sims.shape[1])

    # sort the sims (for local thresholding)
    sorted_idx = np.argsort(sims, axis=1)
    sorted_idx = np.flip(sorted_idx, axis=1)
    
    # last k=max so as to have a complete curve
    ks += (sims.shape[0]-1,)
    
    thresholds = []
    
    # evaluate window sims at various levels of voting, local, and global thresholds
    tot_iter = len(vote_thrs) * len(ks) * len(min_sims)
    with tqdm(total = tot_iter) as pbar:
        for vote_thr in vote_thrs:
            for k in ks:

                # using k, construct the matrix of corr votes
                votes = calc_votes_thr(sims, sorted_idx, k)

                for min_sim in min_sims:
                    
                    # filter out votes below min (if supplied)
                    if min_sim is not None:
                        votes[sims < min_sim] = 0

                    # compute metrics from votes
                    corr_pred = tally_votes(votes, vote_thr)
                    tp, tn, fp, fn = calc_metrics(corr_true, corr_pred)

                    fpr.append((fp / (fp + tn)) if fp+tn > 0 else 0.)
                    tpr.append((tp / (tp + fn)) if tp+fn > 0 else 0.)
                    cm.append((tp, tn, fp, fn))
                    
                    thresholds.append((vote_thr, k, min_sim))
                    
                    pbar.update(1)

    # sort by FPR
    idx = np.argsort(fpr)
    fpr = np.array(fpr)[idx]
    tpr = np.array(tpr)[idx]
    cm = np.array(cm)[idx]
    thresholds = np.array(thresholds)[idx]

    # filter out non-useful thresholds
    if not keep_intermediate:
        keep_idx = []
        for i in range(1,len(fpr)):
            if tpr[i] > tpr[i-1]:
                keep_idx.append(i)
        fpr = fpr[keep_idx]
        tpr = tpr[keep_idx]
        cm = cm[keep_idx]
        thresholds = thresholds[keep_idx]
    
    return fpr, tpr, cm, thresholds


def parse_args():
    parser = argparse.ArgumentParser(
                        prog = 'benchmark-thr.py',
                        description = 'Evaluate FEN correlation performance using thresholding mechanisms.',
                        #epilog = 'Text at the bottom of help'
                        )

    # experiment configuration options
    parser.add_argument('--dists_file', 
                        default = '/data/path2', 
                        type = str,
                        help = "Load the pickle filepath containing the pre-calculated similarity matrix.", 
                        required=True)
    parser.add_argument('--results_file', 
                        default = '/data/path2', 
                        type = str,
                        help = "Path where to save the results (as a pickle file).", 
                        required=True)
    parser.add_argument('--max_windows',
                        default=32,
                        type=int,
                        help = "Set the limit of windows to consider. \
                            If window count is about the max then windows are evenly sampled from the sequence.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # create dir path to results file if necessary
    dirpath = os.path.dirname(args.results_file)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # load previously generated distance matrices
    with open(args.dists_file, 'rb') as fi:
        data = pickle.load(fi)
    va_sims = data['va_sims'].astype(np.float16)
    te_sims = data['te_sims'].astype(np.float16)
    del data
    
    # reduce window count by evenly sampling windows
    if va_sims.shape[-1] > args.max_windows:
        window_idx = np.linspace(0, va_sims.shape[-1], 
                                 args.max_windows, 
                                 endpoint = False,
                                 dtype = int)
        va_sims = va_sims[:,:,window_idx]
        te_sims = te_sims[:,:,window_idx]

    # find min sim threshold using val. set
    min_sims = []
    corr_true = np.eye(va_sims.shape[0], va_sims.shape[1]).astype(bool)
    for f in (0., 0.1, 0.2):
        min_sims.append(calc_min_sim(va_sims, corr_true, f))

    # evaluate performance on test set
    fpr, tpr, cm, thresholds = evaluate(te_sims, min_sims=min_sims)
    tpr = np.concatenate(([0.],tpr,[1.]))
    fpr = np.concatenate(([0.],fpr,[1.]))
    acc = (cm[0][0] + cm[0][1]) / sum(cm[0])
    roc_auc = metrics.auc(fpr, tpr)
    print(f"Test accuracy: {acc}, roc: {roc_auc}")

    # store results
    with open(args.results_file, 'wb') as fi:
        pickle.dump({'fpr': fpr, 
                     'tpr': tpr,
                     'cm': cm,
                     'thresholds': thresholds}, fi)
    
    # plot ROC curve
    import matplotlib.pyplot as plt
    plt.title(f'Receiver Operating Characteristic')
    plt.plot(fpr[1:-1], tpr[1:-1], 
             'b', label = 'AUC = %0.6f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot(np.linspace(0, 1, 100000), 
             np.linspace(0, 1, 100000), 
             'k--')
    plt.xlim([1e-8, 1])
    plt.ylim([-0.03, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xscale('log')
    plt.savefig(os.path.join(dirpath, 'roc.pdf'))

