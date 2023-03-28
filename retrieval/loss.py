import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, features1, features2, logit_scale):
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        
        logits1 = logit_scale * features1 @ features2.T
        
        logits2 = logits1.T
        
        labels = torch.arange(len(logits1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits1, labels) + self.loss_function(logits2, labels))/2

        return loss  
 

