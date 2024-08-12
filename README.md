# ESPRESSO: Enhanced Sequential Processing for Reliable End-to-end Signal Synchronization and Observation 

### An improved traffic correlation method for end-to-end flow correlation attacks against Tor

This repository has the following features:
1. Re-implementation of the DeepCoFFEA correlation method for Tor traffic flows.
2. New Online Triplet mining strategies that can be used in place of Offline mining.
3. New MLP-based correlation prediction that can be used in place of threshold-based window voting.
4. New ESPRESSO transformer-based feature extraction network (FEN) that does not use pre-processed windows.

### USAGE

- Train using the original DeepCoFFEA method and FEN
```
python train.py --loss_margin 0.1 --data_dir ${path/to/rootdir} --dcf 
```

- Train using the ESPRESSO method and FEN
```
python train.py --loss_margin 0.1 --data_dir ${path/to/rootdir} --online --hard
```

- Generate window similarity matrix using a trained FEN
```
python calc-sims.py --dists_file ./exps/1/dists.pkl --ckpt ${path/to/ckptfile} --data_dir ${path/to/rootdir}
```

- Evaluate correlation efficacy using window voting and local thresholding
```
python benchmark-thr.py --dists_file ./exps/1/dists.pkl --results_file ./exps/1/res.pkl
```

- Evaluate correlation efficacy using MLP predictor
```
python benchmark-mlp.py --dists_file ./exps/1/dists.pkl --results_file ./exps/1/res.pkl
```

