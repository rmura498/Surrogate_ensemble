import torch
import torch.nn as nn


class CWLoss(nn.Module):

    def __init__(self, margin=200, targeted=True):
        super(CWLoss, self).__init__()
        self.margin = margin
        self.targeted = targeted

    def forward(self, logits, target_label):

        device = logits.device
        k = torch.tensor(self.margin).float().to(device)
        target_label = target_label.squeeze()
        logits = logits.squeeze()
        onehot_logits = torch.zeros_like(logits)
        onehot_logits[target_label] = logits[target_label]
        other_logits = logits - onehot_logits
        best_other_logit = torch.max(other_logits)
        tgt_label_logit = logits[target_label]

        if self.targeted:
            loss = torch.max(best_other_logit - tgt_label_logit, -k)
        else:
            loss = torch.max(tgt_label_logit - best_other_logit, -k)

        return loss
