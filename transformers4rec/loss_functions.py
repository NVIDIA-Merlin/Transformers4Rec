#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Code adapted from : https://github.com/hungthanhpham94/GRU4REC-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

########
#   New losses: Handle the case where a session has a multiple labels
########


class BPR(nn.Module):
    def __init__(self):
        super(BPR, self).__init__()

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


class BPR_max(nn.Module):
    def __init__(self):
        super(BPR_max, self).__init__()

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
        logit_softmax = torch.softmax(logits_scores, dim=1)
        loss = logit_softmax * loss
        # set to zeros the difference scores of positive targets of the same session
        loss = loss.masked_fill(positive_mask, 0)
        # Average over the nb of negative sample per
        loss = torch.log(loss.sum(1))
        return -torch.mean(loss)


class BPR_max_reg(nn.Module):
    def __init__(self, lambda_):
        super(BPR_max_reg, self).__init__()
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
        logit_softmax = torch.softmax(logits_scores, dim=1)
        loss = logit_softmax * loss
        reg = logit_softmax * (logits_scores ** 2)
        # set to zeros the diff scores of positive targets present in the same session
        loss = loss.masked_fill(positive_mask, 0)
        reg = reg.masked_fill(positive_mask, 0)
        # Average over the nb of negative sample per
        loss = -torch.log(loss.sum(1)) + self.lambda_ * reg.sum(1)
        return torch.mean(loss)


class TOP1(nn.Module):
    def __init__(self):
        super(TOP1, self).__init__()

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


class TOP1_max(nn.Module):
    def __init__(self):
        super(TOP1_max, self).__init__()

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
        logit_softmax = F.softmax(logits_scores, dim=1)
        loss = logit_softmax * loss
        # set to zeros the difference scores of positive targets of the same session
        loss = loss.masked_fill(positive_mask, 0)
        # Average over the nb of negative sample per
        loss = loss.sum(1) / negative_mask.sum(1)
        return torch.mean(loss)
