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


def calc_votes_thr(sims, sorted_sims, k=1):
    """
    """
    local_thr = sorted_sims[:,k,:]
    return (sims >= local_thr).astype(int)


def tally_votes(votes, perc=0.8):
    """
    """
    window_count = votes.shape[-1]
    required_votes = math.ceil(perc * window_count)

    votes = np.sum(votes, axis=-1)
    corr_pred = (votes >= required_votes).astype(int)

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

def filter_votes(sims, votes, min_sim):
    """
    """
    votes[sims < min_sim] = 0
    return votes


def evaluate(sims, 
        ks = (1,2,4,8,16,32,64,128,256,512), 
        min_sim = None,
        vote_thr = 0.8,
        ):
    """
    """
    fpr, tpr, perf_metrics = [], [], []
    
    # correlated when i=j
    corr_true = np.eye(sims.shape[0], sims.shape[1])

    # sort the sims (for local thresholding)
    sorted_idx = np.argsort(sims, axis=1)
    sorted_idx = np.flip(sorted_idx, axis=1)
    sorted_sims = np.take_along_axis(sims, sorted_idx, axis=1)
    
    # last k=max so as to have a complete curve
    ks += (sims.shape[0]-1,)
    
    # evaluate window sims at various levels of k
    for k in tqdm(ks):
        
        # using k, construct the matrix of corr votes
        votes = calc_votes_thr(sims, sorted_sims, k)
        # filter out votes below min (if supplied)
        if min_sim is not None:
            votes = filter_votes(sims, votes, min_sim)

        # compute metrics from votes
        corr_pred = tally_votes(votes, vote_thr)
        tp, tn, fp, fn = calc_metrics(corr_true, corr_pred)
        print(tp, tn, fp, fn, tp+fn, tn+fp)

        fpr.append((fp / (fp + tn)) if fp+tn > 0 else 0.)
        tpr.append((tp / (tp + fn)) if tp+fn > 0 else 0.)
        perf_metrics.append((tp, tn, fp, fn))

    idx = np.argsort(fpr)
    fpr = np.array(fpr)[idx]
    tpr = np.array(tpr)[idx]
    perf_metrics = np.array(perf_metrics)[idx]
    return fpr, tpr, perf_metrics


def parse_args():
    parser = argparse.ArgumentParser(
                        #prog = 'WF Benchmark',
                        #description = 'Train & evaluate WF attack model.',
                        #epilog = 'Text at the bottom of help'
                        )

    # experiment configuration options
    parser.add_argument('--dists_file', 
                        default = '/data/path2', 
                        type = str,
                        help = "Path to dataset root 'pathx' directory.", 
                        required=True)
    parser.add_argument('--results_file', 
                        default = '/data/path2', 
                        type = str,
                        help = "Path to dataset root 'pathx' directory.", 
                        required=True)
    parser.add_argument('--min_sim',
                        default = None,
                        type = float,
                        help = "Percentile threshold of val. sims to use for filtering.")

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

    # find min sim threshold using val. set
    va_sims = data['va_sims']
    if args.min_sim is not None:
        min_sim = calc_min_sim(va_sims, 
                np.eye(va_sims.shape[0], va_sims.shape[1]).astype(bool), 
                args.min_sim)
    else:
        min_sim = None

    # evaluate performance on test set
    te_sims = data['te_sims']
    fpr, tpr, perf_metrics = evaluate(te_sims, min_sim = min_sim)
    acc = (perf_metrics[0][0] + perf_metrics[0][1]) / sum(perf_metrics[0])
    roc_auc = metrics.auc(fpr, tpr)
    print(f"Test accuracy: {acc}, roc: {roc_auc}")

    # store results
    with open(args.results_file, 'wb') as fi:
        pickle.dump({'fpr': fpr, 
                     'tpr': tpr}, fi)
    
    # plot ROC curve
    import matplotlib.pyplot as plt
    plt.title(f'Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.6f' % roc_auc)
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

