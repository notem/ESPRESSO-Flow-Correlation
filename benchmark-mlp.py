import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn import metrics
from tqdm import tqdm
import argparse
import os
import pickle
import transformers


class MyDataset(Dataset):
    def __init__(self, sims):
        self.indices = np.triu_indices(sims.shape[0], m=sims.shape[1])
        self.inputs = sims
        self.targets = np.eye(sims.shape[0], sims.shape[1])

        corr_count = np.sum(np.triu(self.targets))
        corr_weight = len(self.indices) / corr_count
        self.weights = torch.DoubleTensor([corr_weight if self.targets[self.indices[0][i], self.indices[0][i]] 
                                                    else 1-corr_weight for i in range(len(self.indices[0]))])

    def __len__(self):
        return len(self.indices[0])

    def __getitem__(self, idx):
        i = self.indices[0][idx], self.indices[1][idx]
        return (torch.tensor(self.inputs[i], dtype=torch.float), 
                torch.tensor(self.targets[i], dtype=torch.float))


class Predictor(nn.Module):
    """
    Simple MLP for binary prediction
    """
    def __init__(self, dim, drop=0.7, ratio=1, layers=2):
        super(Predictor, self).__init__()
        modules = []
        for i in range(layers):
            fc = nn.Sequential(
                    nn.Linear(dim, int(dim*ratio)) if i == 0 else nn.Linear(int(dim*ratio), int(dim*ratio)),
                    nn.GELU(),
                    #nn.BatchNorm1d(dim*ratio),
                    )
            modules.append(fc)

        self.fc_modules = nn.ModuleList(modules)
        self.pred = nn.Linear(int(dim*ratio), 1)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        for i,module in enumerate(self.fc_modules):
            x = self.dropout(module(x))
        x = torch.sigmoid(self.pred(x))
        return x


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

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    dirpath = os.path.dirname(args.results_file)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    with open(args.dists_file, 'rb') as fi:
        data = pickle.load(fi)

    # Create PyTorch datasets
    tr_dataset = MyDataset(data['va_sims'])
    te_dataset = MyDataset(data['te_sims'])
    
    # Create PyTorch dataloaders
    batch_size = 256
    num_epochs = 5
    
    # use weighted sampler to balance training dataset
    tr_sampler = WeightedRandomSampler(tr_dataset.weights, len(tr_dataset.weights))
    tr_loader = DataLoader(tr_dataset, 
            batch_size = batch_size, 
            sampler = tr_sampler,
            pin_memory = True,
            )
    # (no sampler) evaluate on all pairwise cases in test set
    te_loader = DataLoader(te_dataset, 
            batch_size = batch_size, 
            shuffle = False)
    
    # Instantiate the model and move it to GPU if available
    model = Predictor(dim=tr_dataset.inputs.shape[-1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define the loss function and the optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                                len(tr_loader), 
                                                len(tr_loader)*num_epochs)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, targets in tqdm(tr_loader):
            # Move tensors to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
    
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            preds = outputs >= 0.5
            correct_predictions += (preds == targets.unsqueeze(1)).sum().item()
            total_predictions += targets.size(0)
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            running_loss += loss.item()
    
        train_loss = running_loss / len(tr_loader)
        train_acc = correct_predictions / total_predictions
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
    
    # Put the model in evaluation mode
    model.eval()
    
    # Lists to store the model's outputs and the actual targets
    outputs_list = []
    targets_list = []
    
    # Pass the validation data through the model
    with torch.no_grad():
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, targets in tqdm(te_loader):
            # Move tensors to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
    
            # Forward pass
            outputs = model(inputs)
    
            # Store the outputs and targets
            outputs_list.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

            # metrics
            loss = criterion(outputs, targets.unsqueeze(1))
            preds = outputs >= 0.5
            correct_predictions += (preds == targets.unsqueeze(1)).sum().item()
            total_predictions += targets.size(0)
            running_loss += loss.item()

        test_loss = running_loss / len(te_loader)
        test_acc = correct_predictions / total_predictions
        print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # Compute the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(targets_list, outputs_list)
    roc_auc = metrics.auc(fpr, tpr)

    # store results
    with open(args.results_file, 'wb') as fi:
        pickle.dump({'fpr': fpr, 
                     'tpr': tpr}, fi)
    
    import matplotlib.pyplot as plt
    plt.title(f'Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.6f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'--')
    plt.xlim([1e-8, 1e-1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xscale('log')
    plt.savefig(os.path.join(dirpath, 'roc.pdf'))

