# Dataset Structure & Sources

The code in this project has been written to read-in two different dataset structures for end-to-end Tor flow correlation. These are: (1) plaintext files representing singular samples within flat directories named 'inflow' and 'outflow', and (2) pre-packaged pickle files that contain traffic sequence metadata organized into batches with mutually exclusive circuit usage.

Data loading is handled by the `utils/data.py` file. The main runnable scripts use configuration JSON files located in `configs/exps/` for consistent data sample loading between experiments.

## Sources

The pre-packaged pickle files can be downloaded from the following Google drive folder:
* https://drive.google.com/drive/folders/1snY8OGul-TfPfFL0qibUC3zYQANHKFTI?usp=sharing