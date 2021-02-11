"""
Code adapted from : https://github.com/hungthanhpham94/GRU4REC-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to the number of negative samples (N_B  or N_B + N_A)
        """
        # differences between the item scores
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        # final loss
        loss = -torch.mean(F.logsigmoid(diff))
        return loss


class BPR_max(nn.Module):
    def __init__(self):
        super(BPR_max, self).__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to the number of negative samples (N_B  or N_B + N_A)
        """
        logit_softmax = F.softmax(logit, dim=1)
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        loss = -torch.log(torch.mean(logit_softmax * torch.sigmoid(diff)))
        return loss


class BPR_max_reg(nn.Module):
    def __init__(self, lambda_):
        """
        Args:
            lambda_: regularization hyper-parameter of the negative scores : Takes value between 0 and 1
        """
        super(BPR_max_reg, self).__init__()
        self.lambda_ = lambda_

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to the number of negative samples (N_B  or N_B + N_A)
        """
        logit_softmax = F.softmax(logit, dim=1)
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        loss = -torch.log(torch.mean(logit_softmax * torch.sigmoid(diff)))
        # add regularization
        loss += self.lambda_ * torch.mean(logit_softmax * (logit ** 2))
        return loss


class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to the number of negative samples (N_B  or N_B + N_A)
        """
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.sigmoid(diff).mean() + torch.sigmoid(logit ** 2).mean()
        return loss


class TOP1_max(nn.Module):
    def __init__(self):
        super(TOP1_max, self).__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to the number of negative samples (N_B  or N_B + N_A)
        """
        logit_softmax = F.softmax(logit, dim=1)
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.mean(logit_softmax * (torch.sigmoid(diff) + torch.sigmoid(logit ** 2)))
        return loss


class TOP1_max_reg(nn.Module):
    def __init__(self, lambda_):
        """
        Args:
            lambda_: regularization hyper-parameter of the negative scores : Takes value between 0 and 1
        """
        super(TOP1_max_reg, self).__init__()
        self.lambda_ = lambda_

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logit scores for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to the number of negative samples (N_B  or N_B + N_A)
        """
        logit_softmax = F.softmax(logit, dim=1)
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.mean(logit_softmax * (torch.sigmoid(diff) + torch.sigmoid(logit ** 2)))
        # add regularization
        loss += self.lambda_ * torch.mean(logit_softmax * (logit ** 2))
        return loss