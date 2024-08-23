import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
        self.sim = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor, positive, negative):
        pos_sim = self.sim(anchor, positive)
        neg_sim = self.sim(anchor, negative)
        loss = F.relu(neg_sim - pos_sim + self.margin)
        return loss.mean()


def compute_sim(in_emb, out_emb):
    """
    """
    # Normalize each vector (element) to have unit norm
    norms = torch.norm(in_emb, p=2, dim=-1, keepdim=True)  # Compute L2 norms
    in_emb = in_emb / norms  # Divide by norms to normalize
    
    norms = torch.norm(out_emb, p=2, dim=-1, keepdim=True)  # Compute L2 norms
    out_emb = out_emb / norms  # Divide by norms to normalize
    
    # Compute pairwise cosine similarity
    if in_emb.dim() == 2:
        all_sim = torch.mm(in_emb, out_emb.t())
    elif in_emb.dim() == 3:
        all_sim = torch.matmul(in_emb.permute(1,0,2), out_emb.permute(1,2,0))
        all_sim = all_sim.mean(0)

    return all_sim


class OnlineCosineTripletLoss(nn.Module):
    """
    """
    def __init__(self, margin = 0.1, semihard = True):
        super(OnlineCosineTripletLoss, self).__init__()
        self.margin = margin
        self.semihard = semihard

    def _get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True if the triplet (a, p, n) is valid.

        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]

        Args:
            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]
        """
        # Check that i, j, and k are distinct
        indices_equal = torch.eye(labels.size(0)).bool().to(labels.device)
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        labels = labels.unsqueeze(0)
        label_equal = labels == labels.transpose(0, 1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = i_equal_j & (~i_equal_k)

        # Combine the two masks
        mask = distinct_indices & valid_labels

        #return mask
        return valid_labels

    def forward(self, in_embeddings, out_embeddings):
        """
        """
        all_sim = compute_sim(in_embeddings, out_embeddings)

        labels = torch.arange(in_embeddings.size(0)).to(in_embeddings.get_device())

        # mask of valid triplets
        mask = self._get_triplet_mask(labels).float()
        if all_sim.dim() == 3:
            mask = mask.unsqueeze(0)

        # expand dims for pairwise comparison
        anc_pos_sim = all_sim.unsqueeze(-1)
        anc_neg_sim = all_sim.unsqueeze(-2)

        loss = F.relu(anc_neg_sim - anc_pos_sim + self.margin) * mask

        # calculate average loss (disregarding invalid & easy triplets)
        if self.semihard:
            nonzero = torch.count_nonzero(loss)
            if nonzero > 0:
                loss = torch.sum(loss) / nonzero
            else:
                loss = loss.mean() * 0.
        else:
            loss = loss.sum() / mask.sum()
        return loss


class OnlineHardCosineTripletLoss(nn.Module):
    """
    """
    def __init__(self, margin=0.1):
        super(OnlineHardCosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, in_embeddings, out_embeddings):
        """
        Args:
        """
        all_sim = compute_sim(in_embeddings, out_embeddings)

        # find hardest positive pairs (when positive has low sim)
        # mask of all valid positives
        mask_anc_pos = torch.eye(in_embeddings.size(0)).to(in_embeddings.get_device()).bool()
        if all_sim.dim() == 3:
            mask_anc_pos = mask_anc_pos.unsqueeze(0)
        # prevent invalid pos by increasing sim
        anc_pos_sim = all_sim + (~mask_anc_pos * 999).float()
        # select minimum sim positives
        hardest_pos_sim = anc_pos_sim.min(dim=-1)[0]

        # find hardest negative triplets (when negative has high sim)
        # set invalid negatives to 0
        anc_neg_sim = all_sim * ~mask_anc_pos
        # select maximum sim negatives
        hardest_neg_sim = anc_neg_sim.max(dim=-1)[0]

        loss = F.relu(hardest_neg_sim - hardest_pos_sim + self.margin)

        # calculate average loss (disregarding invalid & easy triplets)
        nonzero = torch.count_nonzero(loss)
        if nonzero > 0:
            loss = torch.sum(loss) / nonzero
        else:
            loss = loss.mean() * 0.
        return loss
