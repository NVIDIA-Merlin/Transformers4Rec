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
        logit_softmax = F.softmax(logit, dim = 1)
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
        logit_softmax = F.softmax(logit, dim = 1)
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
        logit_softmax = F.softmax(logit, dim = 1)
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
        logit_softmax = F.softmax(logit, dim = 1)
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.mean(logit_softmax * (torch.sigmoid(diff) + torch.sigmoid(logit ** 2)))
        # add regularization
        loss += self.lambda_ * torch.mean(logit_softmax * (logit ** 2))
        return loss

########
#   New losses: Handle the case where a session has a multiple labels
########


class New_BPR(nn.Module):
    def __init__(self):
        super(New_BPR, self).__init__()

    def forward(self, logits_scores, negative_mask):
        """
        Args:
            logits_scores: (#pos_target, #neg_samples):  scores of positive next items for all sessions
                                                            + All negative samples (mini-batch + additional samples )
            negative_mask: (#pos_target, #neg_samples) : specify the negative items for each positive target
        """
        positive_mask = ~negative_mask
        positives = logits_scores.diag().view(-1, 1).expand_as(logits_scores)
        diff = positives - logits_scores
        loss = F.logsigmoid(diff)
        # set to zeros the difference scores of positive targets of the same session
        loss = loss.masked_fill(positive_mask, 0)
        # Average over the nb of negative sample per
        loss = loss.sum(1) / negative_mask.sum(1)
        return -torch.mean(loss)


class New_BPR_max(nn.Module):
    def __init__(self):
        super(New_BPR_max, self).__init__()

    def forward(self, logits_scores, negative_mask):
        """
        Args:
            logits_scores: (#pos_target, #neg_samples):  scores of positive next items for all sessions
                                                            + All negative samples (mini-batch + additional samples )
            negative_mask: (#pos_target, #neg_samples) : specify the negative items for each positive target
        """
        positive_mask = ~negative_mask
        positives = logits_scores.diag().view(-1, 1).expand_as(logits_scores)
        diff = positives - logits_scores
        loss = F.logsigmoid(diff)
        logit_softmax = torch.softmax(logits_scores, dim = 1)
        loss = logit_softmax * loss
        # set to zeros the difference scores of positive targets of the same session
        loss = loss.masked_fill(positive_mask, 0)
        # Average over the nb of negative sample per
        loss = torch.log(loss.sum(1))
        return -torch.mean(loss)


class NewBPR_max_reg(nn.Module):
    def __init__(self, lambda_):
        super(NewBPR_max_reg, self).__init__()
        self.lambda_ = lambda_

    def forward(self, logits_scores, negative_mask):
        """
        Args:
            logits_scores: (#pos_target, #neg_samples):  scores of positive next items for all sessions
                                                            + All negative samples (mini-batch + additional samples )
            negative_mask: (#pos_target, #neg_samples) : specify the negative items for each positive target
        """
        positive_mask = ~negative_mask
        positives = logits_scores.diag().view(-1, 1).expand_as(logits_scores)
        diff = positives - logits_scores
        loss = F.sigmoid(diff)
        logit_softmax = torch.softmax(logits_scores, dim = 1)
        loss = logit_softmax * loss
        reg = logit_softmax * (logits_scores**2)
        # set to zeros the diff scores of positive targets present in the same session
        loss = loss.masked_fill(positive_mask, 0)
        reg = reg.masked_fill(positive_mask, 0)
        # Average over the nb of negative sample per
        loss = -torch.log(loss.sum(1)) + self.lambda_ * reg.sum(1)
        return torch.mean(loss)


class NewTOP1(nn.Module):
    def __init__(self):
        super(NewTOP1, self).__init__()

    def forward(self, logits_scores, negative_mask):
        """
        Args:
            logits_scores: (#pos_target, #neg_samples):  scores of positive next items for all sessions
                                                            + All negative samples (mini-batch + additional samples )
            negative_mask: (#pos_target, #neg_samples) : specify the negative items for each positive target
        """
        positive_mask = ~negative_mask
        positives = logits_scores.diag().view(-1, 1).expand_as(logits_scores)
        diff = positives - logits_scores
        penalization = torch.sigmoid(logits_scores ** 2)
        loss = torch.sigmoid(-diff) + penalization
        # set to zeros the difference scores of positive targets of the same session
        loss = loss.masked_fill(positive_mask, 0)
        # Average over the nb of negative sample per
        loss = loss.sum(1) / negative_mask.sum(1)
        return torch.mean(loss)


class NewTOP1_max(nn.Module):
    def __init__(self):
        super(NewTOP1_max, self).__init__()

    def forward(self, logits_scores, negative_mask):
        """
        Args:
            logits_scores: (#pos_target, #neg_samples):  scores of positive next items for all sessions
                                                            + All negative samples (mini-batch + additional samples )
            negative_mask: (#pos_target, #neg_samples) : specify the subset of negative items for each positive target
        """
        positive_mask = ~negative_mask
        positives = logits_scores.diag().view(-1, 1).expand_as(logits_scores)
        diff = positives - logits_scores
        penalization = torch.sigmoid(logits_scores ** 2)
        loss = torch.sigmoid(-diff) + penalization
        logit_softmax = F.softmax(logits_scores, dim = 1)
        loss = logit_softmax * loss
        # set to zeros the difference scores of positive targets of the same session
        loss = loss.masked_fill(positive_mask, 0)
        # Average over the nb of negative sample per
        loss = loss.sum(1) / negative_mask.sum(1)
        return torch.mean(loss)
