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

import torch

from transformers4rec.torch.tabular.base import TabularBlock


class PostContextFusion(TabularBlock):
    """ "

    This block leverages the Latent Cross [1] technique to provide contextual
    information that is not suitable to be fed as input for the sequential module.
    For example, it might be features about the target item to be predicted
    (In Next Item Prediction task) or some features from other user-level or
    sequence-level features.
    In other terms, contextual information  is merged with the output of
    the sequential module right before the prediction step.
    This technique led to a performance boost in our SIGIR'21 challenge [2]
    and Recsys'22 challenge [Ref TODO] solutions.

    Parameters
    ----------
    sequential_module : Block
        The sequential module that returns a 3-D hidden representation of the raw input sequence.
    post_context_module: Block
        The block that encodes contextual information about all input items, including targets.
        The block can returns a 3-D vector (sequence-level context ) or
         a 2-D vector (user-level context)


    References:
    -----------
    [1] Alex Beutel et al. Latent Cross: Making Use of Context in Recurrent Recommender Systems
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46488.pdf

    [2] Gabriel de Souza P. Moreira et al. Transformers with multi-modal features and
    post-fusion context.
    for e-commerce session-based recommendation
    https://arxiv.org/pdf/2107.05124.pdf

    """

    def __init__(
        self,
        sequential_module,
        post_context_module,
        fusion_aggregation="elementwise-mul",
    ):
        super(PostContextFusion, self).__init__()
        self.sequential_module = sequential_module
        self.post_context_module = post_context_module
        self.fusion_aggregation = fusion_aggregation

        _, seq_length, hidden_dim = sequential_module[-1].output_size()

        post_context_last_dim = post_context_module.output_size()[-1]

        self.seq_length = seq_length
        self.inputs = self.sequential_module.inputs

        if fusion_aggregation.startswith("elementwise"):
            self.last_dim = hidden_dim
            self.context_projection = torch.nn.Linear(post_context_last_dim, hidden_dim)
        elif fusion_aggregation == "concat":
            self.last_dim = hidden_dim + post_context_last_dim

    def forward(self, inputs, training=False, testing=False, **kwargs):
        seq_rep = self.sequential_module(inputs, training=training, testing=testing, **kwargs)
        context_rep = self.post_context_module(inputs, training=training)

        if len(context_rep.size()) == 2:
            # repeat the context vector for each position in the sequence
            context_rep = context_rep.unsqueeze(dim=1).repeat(1, self.sequence_length, 1)

        if self.fusion_aggregation.startswith("elementwise"):
            context_rep = self.context_projection(context_rep)

        if self.fusion_aggregation == "concat":
            output = torch.cat([seq_rep, context_rep], axis=-1)
        elif self.fusion_aggregation == "elementwise-mul":
            output = torch.multiply(seq_rep, 1.0 + context_rep)
        elif self.fusion_aggregation == "elementwise-sum":
            output = seq_rep + context_rep
        else:
            raise ValueError(
                f"The aggregation {self.fusion_aggregation} is not supported,"
                f"please select one of the following aggregations "
                f"['concat', 'elementwise-mul', 'elementwise-sum']"
            )
        return output

    def _get_name(self):
        return "PostContextFusion"

    def output_size(self, input_size=None):
        return [-1, self.seq_length, self.last_dim]

    def forward_output_size(self, input_size):
        return torch.Size(list(input_size[:-1]) + [self.output_size()])
