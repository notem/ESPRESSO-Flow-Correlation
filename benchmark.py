import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn import metrics
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, sims, corr):
        self.indices = np.triu_indices(sims, k=1)
        self.inputs = sims
        self.targets = corr

        corr_count = np.sum(np.triu(self.targets, k=1))
        corr_weight = len(self.indices) / corr_count
        self.weights = torch.DoubleTensor([corr_weight if corr[idx] else 1-corr_weight for idx in self.indices])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return (torch.tensor(self.inputs[idx], dtype=torch.float), 
                torch.tensor(self.targets[idx], dtype=torch.float))


class Predictor(nn.Module):
    def __init__(self, dim, drop=0.7, ratio=0.2, layers=3):
        """
        """
        super(Predictor, self).__init__()
        modules = []
        for i in range(layers):
            fc = nn.Sequential(
                    nn.Linear(dim, int(dim*ratio))) if i == 0 else nn.Linear(int(dim*ratio), int(dim*ratio)),
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
    parser.add_argument('--name', 
                        default = '/data/path2', 
                        type = str,
                        help = "Path to dataset root 'pathx' directory.", 
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    with open(args.dists_file, 'rb') as fi:
        data = pickle.load(fi)

    # Create PyTorch datasets
    tr_dataset = MyDataset(data['va_sims'], data['va_corr'])
    te_dataset = MyDataset(data['te_sims'], data['te_corr'])
    
    # Create PyTorch dataloaders
    batch_size = 256
    tr_sampler = WeightedRandomSampler(tr_dataset.weights, len(tr_dataset.weights))
    tr_loader = DataLoader(tr_dataset, 
            batch_size = batch_size, 
            sampler = tr_sampler,
            pin_memory = True,
            shuffle = True)
    te_loader = DataLoader(te_dataset, 
            batch_size = batch_size, 
            shuffle = False)
    
    
    # Instantiate the model and move it to GPU if available
    model = Predictor(dim=val_inputs.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    num_epochs = 10
    
    # Define the loss function and the optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                                len(train_loader)*10, 
                                                len(train_loader)*num_epochs)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, targets in tqdm(train_loader):
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
    
        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions / total_predictions
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
    
    # Put the model in evaluation mode
    model.eval()
    
    # Lists to store the model's outputs and the actual targets
    outputs_list = []
    targets_list = []
    
    # Pass the validation data through the model
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            #outputs_list.extend(inputs.cpu().numpy())
    
            # Move tensors to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
    
            # Forward pass
            outputs = model(inputs)
    
            # Store the outputs and targets
            outputs_list.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    
    # Compute the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(targets_list, outputs_list)
    roc_auc = metrics.auc(fpr, tpr)
    
    import matplotlib.pyplot as plt
    plt.title(f'Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.6f' % roc_auc)
    plt.legend(loc = 'lower right')
    #plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.000001, .1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xscale('log')
    plt.savefig('roc.pdf')

