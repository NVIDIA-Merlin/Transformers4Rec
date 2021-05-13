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
A meta class supports various (Huggingface) transformer models for RecSys tasks.

"""

import logging
import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from transformers import ElectraModel, GPT2Model, PreTrainedModel, XLNetModel

from .loss_functions import BPR, TOP1, BPR_max, BPR_max_reg, TOP1_max
from .recsys_tasks import RecSysTask

logger = logging.getLogger(__name__)

torch.manual_seed(0)


class AttnMerge(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W1 = nn.ModuleList([nn.Linear(input_dim, input_dim)] * output_dim)
        self.output_dim = output_dim
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inp):
        out = []
        for i in range(self.output_dim):
            attn_weight = self.softmax(self.W1[i](inp))
            out.append(torch.mul(attn_weight, inp).sum(-1))
        return torch.stack(out, dim=-1)


class ProjectionNetwork(nn.Module):
    """
    Project item interaction embeddings into model's hidden size
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        inp_merge,
        layer_norm_all_features,
        input_dropout,
        tf_out_act,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inp_merge = inp_merge
        self.layer_norm_all_features = layer_norm_all_features
        self.input_dropout = input_dropout
        self.tf_out_act = tf_out_act
        self.input_dropout = nn.Dropout(self.input_dropout)
        if self.layer_norm_all_features:
            self.layer_norm_all_input = nn.LayerNorm(normalized_shape=self.input_dim)
        if self.inp_merge == "mlp":
            self.merge = nn.Linear(self.input_dim, output_dim)

        elif self.inp_merge == "attn":
            self.merge = AttnMerge(self.input_dim, output_dim)

        elif self.inp_merge == "identity":
            assert self.input_dim == self.output_dim, (
                "Input dim '%s' should be equal to the model's hidden size '%s' when inp_merge=='identity'"
                % (self.input_dim, self.output_dim)
            )
            self.merge = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, inp):
        if self.inp_merge == "mlp" and self.layer_norm_all_features:
            return self.tf_out_act(
                self.merge(self.layer_norm_all_input(self.input_dropout(inp)))
            )
        elif self.inp_merge == "mlp":
            return self.tf_out_act(self.merge(self.input_dropout(inp)))
        return self.merge(inp)


def get_embedding_size_from_cardinality(cardinality, multiplier=2.0):
    # A rule-of-thumb from Google.
    embedding_size = int(math.ceil(math.pow(cardinality, 0.25) * multiplier))
    return embedding_size


class RecSysMetaModel(PreTrainedModel):
    """
    vocab_sizes : sizes of vocab for each discrete inputs
        e.g., [product_id_vocabs, category_vocabs, etc.]
    """

    def __init__(self, model, config, model_args, data_args, feature_map):
        super(RecSysMetaModel, self).__init__(config)

        # Args for Replacement Token Detection
        self.rtd = model_args.rtd
        self.rtd_tied_generator = model_args.rtd_tied_generator
        self.rtd_generator_loss_weight = model_args.rtd_generator_loss_weight
        self.rtd_discriminator_loss_weight = model_args.rtd_discriminator_loss_weight
        self.rtd_sample_from_batch = model_args.rtd_sample_from_batch
        self.rtd_use_batch_interaction = model_args.rtd_use_batch_interaction

        self.layer_norm_featurewise = model_args.layer_norm_featurewise
        self.layer_norm_all_features = model_args.layer_norm_all_features

        self.items_ids_sorted_by_freq = None
        self.neg_samples = None

        # Load model
        if self.rtd:
            # Two electra models are needed for RTD task
            self.model, self.discriminator = model
        else:
            self.model = model

        self.feature_map = feature_map

        self.pad_token = data_args.pad_token
        self.mask_token = data_args.mask_token
        self.session_aware = data_args.session_aware
        self.session_aware_features_prefix = data_args.session_aware_features_prefix

        self.use_ohe_item_ids_inputs = model_args.use_ohe_item_ids_inputs
        self.stochastic_shared_embeddings_replacement_prob = (
            model_args.stochastic_shared_embeddings_replacement_prob
        )

        self.loss_scale_factor = model_args.loss_scale_factor
        self.softmax_temperature = model_args.softmax_temperature
        self.label_smoothing = model_args.label_smoothing

        self.mf_constrained_embeddings = model_args.mf_constrained_embeddings
        self.item_id_embeddings_init_std = model_args.item_id_embeddings_init_std
        self.other_embeddings_init_std = model_args.other_embeddings_init_std

        self.item_embedding_dim = model_args.item_embedding_dim
        self.features_same_size_item_embedding = (
            model_args.features_same_size_item_embedding
        )
        self.embedding_dim_from_cardinality_multiplier = (
            model_args.embedding_dim_from_cardinality_multiplier
        )

        self.numeric_features_project_to_embedding_dim = (
            model_args.numeric_features_project_to_embedding_dim
        )
        self.numeric_features_soft_one_hot_encoding_num_embeddings = (
            model_args.numeric_features_soft_one_hot_encoding_num_embeddings
        )

        if (
            self.numeric_features_soft_one_hot_encoding_num_embeddings > 0
            and self.numeric_features_project_to_embedding_dim == 0
        ):
            raise ValueError(
                "You must set --numeric_features_project_to_embedding_dim to a value greater than zero when using Soft One-Hot Encoding (--numeric_features_soft_one_hot_encoding_num_embeddings)"
            )

        self.input_features_aggregation = model_args.input_features_aggregation

        self.negative_sampling = model_args.negative_sampling

        self.total_seq_length = data_args.total_seq_length

        self.define_features_layers(model_args)

        self.neg_sampling_store_size = model_args.neg_sampling_store_size
        self.neg_sampling_extra_samples_per_batch = (
            model_args.neg_sampling_extra_samples_per_batch
        )
        self.neg_sampling_alpha = model_args.neg_sampling_alpha

        self.inp_merge = model_args.inp_merge
        if self.inp_merge == "identity":
            if self.rtd and not self.rtd_tied_generator:
                raise Exception(
                    "When using --rtd and --rtd_tied_generator, the --inp_merge cannot be 'identity'"
                )
        if model_args.tf_out_activation == "tanh":
            self.tf_out_act = torch.tanh
        elif model_args.tf_out_activation == "relu":
            self.tf_out_act = torch.relu

        self.merge = ProjectionNetwork(
            self.input_combined_dim,
            config.hidden_size,
            self.inp_merge,
            self.layer_norm_all_features,
            model_args.input_dropout,
            self.tf_out_act,
        )

        if not self.rtd_tied_generator:
            self.merge_disc = ProjectionNetwork(
                self.input_combined_dim,
                model_args.d_model,
                self.inp_merge,
                self.layer_norm_all_features,
                model_args.input_dropout,
                self.tf_out_act,
            )

        self.eval_on_last_item_seq_only = model_args.eval_on_last_item_seq_only
        self.train_on_last_item_seq_only = model_args.train_on_last_item_seq_only

        self.n_layer = model_args.n_layer

        # Args for Masked-LM task
        self.mlm = model_args.mlm
        self.mlm_probability = model_args.mlm_probability

        # Args for Permuted-LM task
        self.plm = model_args.plm
        self.plm_max_span_length = model_args.plm_max_span_length
        self.plm_probability = model_args.plm_probability
        self.plm_mask_input = model_args.plm_mask_input
        self.plm_permute_all = model_args.plm_permute_all

        # Creating a trainable embedding for masking inputs for Masked LM
        self.masked_item_embedding = nn.Parameter(torch.Tensor(config.hidden_size)).to(
            self.device
        )
        nn.init.normal_(
            self.masked_item_embedding,
            mean=0,
            std=0.001,
        )

        self.similarity_type = model_args.similarity_type
        self.margin_loss = model_args.margin_loss

        self.output_layer = nn.Linear(config.hidden_size, self.target_dim).to(
            self.device
        )

        self.loss_type = model_args.loss_type
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.output_layer_bias = nn.Parameter(torch.Tensor(self.target_dim)).to(
            self.device
        )
        nn.init.zeros_(self.output_layer_bias)

        # create prediction module for electra discriminator: Two dense layers
        self.dense_discriminator = nn.Linear(model_args.d_model, model_args.d_model)
        self.discriminator_prediction = nn.Linear(model_args.d_model, 1)

        if self.label_smoothing > 0.0:
            self.loss_nll = LabelSmoothCrossEntropyLoss(smoothing=self.label_smoothing)
        else:
            self.loss_nll = nn.NLLLoss(ignore_index=self.pad_token)

        if self.loss_type == "top1":
            self.loss_fn = TOP1()

        elif self.loss_type == "top1_max":
            self.loss_fn = TOP1_max()

        elif self.loss_type == "bpr":
            self.loss_fn = BPR()

        # elif self.loss_type == 'bpr_max':
        #    self.loss_fn = BPR_max()

        elif self.loss_type == "bpr_max_reg":
            self.loss_fn = BPR_max_reg(lambda_=model_args.bpr_max_reg_lambda)

        elif self.loss_type != "cross_entropy":
            raise NotImplementedError

        if model_args.model_type == "reformer":
            tf_out_size = model_args.d_model * 2
        elif self.rtd and not self.rtd_tied_generator:
            tf_out_size = config.hidden_size
        else:
            tf_out_size = model_args.d_model

        if model_args.mf_constrained_embeddings:
            transformer_output_projection_dim = self.item_embedding_dim

        else:
            transformer_output_projection_dim = config.hidden_size

        self.transformer_output_project = nn.Linear(
            tf_out_size, transformer_output_projection_dim
        ).to(self.device)

        if self.similarity_type in ["concat_mlp", "multi_mlp"]:
            m_factor = 2 if self.similarity_type == "concat_mlp" else 1
            self.sim_mlp = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "linear0",
                            nn.Linear(
                                model_args.d_model * m_factor, model_args.d_model
                            ).to(self.device),
                        ),
                        ("relu0", nn.LeakyReLU()),
                        (
                            "linear1",
                            nn.Linear(model_args.d_model, model_args.d_model // 2).to(
                                self.device
                            ),
                        ),
                        ("relu1", nn.LeakyReLU()),
                        (
                            "linear2",
                            nn.Linear(
                                model_args.d_model // 2, model_args.d_model // 4
                            ).to(self.device),
                        ),
                        ("relu2", nn.LeakyReLU()),
                        (
                            "linear3",
                            nn.Linear(model_args.d_model // 4, 1).to(self.device),
                        ),
                        ("sigmoid", nn.Sigmoid()),
                    ]
                )
            )

    def forward(self, *args, **kwargs):
        inputs = kwargs

        # Step1. Unpack inputs, get embedding, and concatenate them
        label_seq = None
        (pos_inp, label_seq, metadata_for_pred_logging) = self.feature_process(inputs)

        assert label_seq is not None, "label sequence is not declared in feature_map"

        # Define pre-training task class
        recsys_task = RecSysTask(self.pad_token, self.device, self.training)

        # To mark past sequence labels
        if self.session_aware:
            masked_past_session = torch.zeros_like(
                label_seq, dtype=torch.long, device=self.device
            )
        if self.mlm:
            """
            Masked Language Model
            """
            label_seq_trg, label_mlm_mask = recsys_task.mask_tokens(
                label_seq, self.mlm_probability
            )

            # To mark past sequence labels
            if self.session_aware:
                label_seq_trg = torch.cat([masked_past_session, label_seq_trg], axis=1)
                label_mlm_mask = torch.cat(
                    [masked_past_session.bool(), label_mlm_mask], axis=1
                )
        elif self.plm:
            """
            Permutation Language Model
            """
            (
                label_seq_trg,
                label_plm_mask,
                target_mapping,
                perm_mask,
            ) = recsys_task.plm_mask_tokens(
                label_seq,
                max_span_length=self.plm_max_span_length,
                plm_probability=self.plm_probability,
                plm_permute_all=self.plm_permute_all,
            )
            # To mark past sequence labels
            if self.session_aware:
                label_seq_trg = torch.cat([masked_past_session, label_seq_trg], axis=1)
                label_plm_mask = torch.cat(
                    [masked_past_session.bool(), label_plm_mask], axis=1
                )

        else:
            """
            Causal Language Modeling - Predict Next token
            """

            label_seq_inp = label_seq[:, :-1]
            label_seq_trg = label_seq[:, 1:]

            # As after shifting the sequence length will be subtracted by one, adding a masked item in
            # the sequence to return to the initial sequence. This is important for ReformerModel(), for example
            label_seq_inp = torch.cat(
                [
                    label_seq_inp,
                    torch.zeros(
                        (label_seq_inp.shape[0], 1), dtype=label_seq_inp.dtype
                    ).to(self.device),
                ],
                axis=-1,
            )
            label_seq_trg = torch.cat(
                [
                    label_seq_trg,
                    torch.zeros(
                        (label_seq_trg.shape[0], 1), dtype=label_seq_trg.dtype
                    ).to(self.device),
                ],
                axis=-1,
            )

            # apply mask on input where target is on padding token
            mask_trg_pad = label_seq_trg != self.pad_token

            label_seq_inp = label_seq_inp * mask_trg_pad

            # When evaluating, computes metrics only for the last item of the session
            if (self.eval_on_last_item_seq_only and not self.training) or (
                self.train_on_last_item_seq_only and self.training
            ):
                rows_ids = torch.arange(
                    label_seq_inp.size(0), dtype=torch.long, device=self.device
                )
                last_item_sessions = mask_trg_pad.sum(axis=1) - 1
                label_seq_trg_eval = torch.zeros(
                    label_seq_trg.shape, dtype=torch.long, device=self.device
                )
                label_seq_trg_eval[rows_ids, last_item_sessions] = label_seq_trg[
                    rows_ids, last_item_sessions
                ]
                # Updating labels and mask
                label_seq_trg = label_seq_trg_eval
                mask_trg_pad = label_seq_trg != self.pad_token

            # To mark past sequence labels
            if self.session_aware:
                label_seq_trg_original = label_seq_trg.clone()
                label_seq_trg = torch.cat([masked_past_session, label_seq_trg], axis=1)
                mask_trg_pad = torch.cat([masked_past_session, mask_trg_pad], axis=1)

        # Creating an additional feature with the position in the sequence
        metadata_for_pred_logging["seq_pos"] = torch.arange(
            1, label_seq.shape[1] + 1, device=self.device
        ).repeat(label_seq.shape[0], 1)
        metadata_for_pred_logging["seq_len"] = (
            (label_seq != self.pad_token)
            .int()
            .sum(axis=1)
            .unsqueeze(-1)
            .repeat(1, label_seq.shape[1])
        )
        # Keeping only metadata features for the next-clicks (targets)
        if not (self.mlm and self.training) and not (self.plm and self.training):
            for feat_name in metadata_for_pred_logging:
                metadata_for_pred_logging[feat_name] = metadata_for_pred_logging[
                    feat_name
                ][:, 1:]

                # As after shifting the sequence length will be subtracted by one, adding a masked item in
                # the sequence to return to the initial sequence. This is important for ReformerModel(), for example
                metadata_for_pred_logging[feat_name] = torch.cat(
                    [
                        metadata_for_pred_logging[feat_name],
                        torch.zeros(
                            (metadata_for_pred_logging[feat_name].shape[0], 1),
                            dtype=metadata_for_pred_logging[feat_name].dtype,
                        ).to(self.device),
                    ],
                    axis=-1,
                )

        # Step 2. Merge features
        pos_emb = self.merge(pos_inp)

        if self.mlm:
            # Masking inputs (with trainable [mask] embedding]) at masked label positions
            pos_emb_inp = torch.where(
                label_mlm_mask.unsqueeze(-1).bool(),
                self.masked_item_embedding.to(pos_emb.dtype),
                pos_emb,
            )
        elif self.plm:
            # The permutation attention mask will prevent to leak information about the masked item to predict
            # So no need to corrupt input with masked token:
            # Similar to the original XLNET tf implementation
            if not self.plm_mask_input:
                pos_emb_inp = pos_emb
            # Masking span-based prediction inputs (with trainable [mask] embedding]) at masked label positions:
            # Similar to HF implementation
            else:
                pos_emb_inp = torch.where(
                    label_plm_mask.unsqueeze(-1).bool(),
                    self.masked_item_embedding.to(pos_emb.dtype),
                    pos_emb,
                )
        else:
            # Truncating the input sequences length to -1
            pos_emb_inp = pos_emb[:, :-1]

            # As after shifting the sequence length will be subtracted by one, adding a masked item in
            # the sequence to return to the initial sequence. This is important for ReformerModel(), for example
            pos_emb_inp = torch.cat(
                [
                    pos_emb_inp,
                    torch.zeros(
                        (pos_emb_inp.shape[0], 1, pos_emb_inp.shape[2]),
                        dtype=pos_emb_inp.dtype,
                    ).to(self.device),
                ],
                axis=1,
            )

            # Replacing the inputs corresponding to masked label with a trainable embedding
            pos_emb_inp = torch.where(
                mask_trg_pad.unsqueeze(-1).bool(),
                pos_emb_inp,
                self.masked_item_embedding.to(pos_emb_inp.dtype),
            )

        # Step3. Run forward pass on model architecture

        if not isinstance(self.model, PreTrainedModel):  # Checks if its a transformer
            # compute output through RNNs
            results = self.model(input=pos_emb_inp)

            if type(results) is tuple or type(results) is list:
                pos_emb_pred = results[0]
            else:
                pos_emb_pred = results

            model_outputs = (None,)

        else:
            """
            Transformer Models
            """

            if type(self.model) is GPT2Model:
                seq_len = pos_emb_inp.shape[1]
                # head_mask has shape n_layer x batch x n_heads x N x N
                head_mask = (
                    torch.tril(
                        torch.ones(
                            (seq_len, seq_len), dtype=torch.uint8, device=self.device
                        )
                    )
                    .view(1, 1, 1, seq_len, seq_len)
                    .repeat(self.n_layer, 1, 1, 1, 1)
                )

                model_outputs = self.model(
                    inputs_embeds=pos_emb_inp,
                    head_mask=head_mask,
                )

            elif self.plm:
                assert (
                    type(self.model) is XLNetModel
                ), "Permutation language model is only supported for XLNET model "
                model_outputs = self.model(
                    inputs_embeds=pos_emb_inp,
                    target_mapping=target_mapping,
                    perm_mask=perm_mask,
                )

            elif self.rtd:
                assert (
                    type(self.model) is ElectraModel
                ), "Replacement token detection is only supported for ELECTRA model"
                model_outputs = self.model(inputs_embeds=pos_emb_inp)

            else:
                model_outputs = self.model(inputs_embeds=pos_emb_inp)

            pos_emb_pred = model_outputs[0]
            model_outputs = tuple(model_outputs[1:])

        pos_emb_pred = self.tf_out_act(self.transformer_output_project(pos_emb_pred))

        trg_flat = label_seq_trg.flatten()
        non_pad_mask = trg_flat != self.pad_token

        labels_all = torch.masked_select(trg_flat, non_pad_mask)

        # Step4. Compute logit and label for neg+pos samples

        # remove zero padding elements
        pos_emb_pred = self.remove_pad_3d(pos_emb_pred, non_pad_mask)

        if not self.mlm and not self.plm:

            if self.session_aware:
                non_pad_original_mask = (
                    label_seq_trg_original.flatten() != self.pad_token
                )
                for feat_name in metadata_for_pred_logging:
                    metadata_for_pred_logging[feat_name] = torch.masked_select(
                        metadata_for_pred_logging[feat_name].flatten(),
                        non_pad_original_mask,
                    )
            else:
                # Keeping removing zero-padded items metadata features for the next-clicks (targets), so that they are aligned
                for feat_name in metadata_for_pred_logging:
                    metadata_for_pred_logging[feat_name] = torch.masked_select(
                        metadata_for_pred_logging[feat_name].flatten(), non_pad_mask
                    )

        if self.mf_constrained_embeddings:

            logits_all = F.linear(
                pos_emb_pred,
                weight=self.embedding_tables[self.label_embedding_table_name].weight,
                bias=self.output_layer_bias,
            )
        else:
            logits_all = self.output_layer(pos_emb_pred)

        # Softmax temperature to reduce model overconfidence and better calibrate probs and accuracy
        logits_all = torch.div(logits_all, self.softmax_temperature)

        if not self.negative_sampling:
            predictions_all = self.log_softmax(logits_all)
            loss_ce = self.loss_nll(predictions_all, labels_all)
            loss = loss_ce
            # accuracy
            # _, max_idx = torch.max(logits_all, dim=1)
            # train_acc = (max_idx == labels_all).mean(dtype = torch.float32)

        if self.negative_sampling:
            # Compute pairwise loss using negative samples
            # The negative samples are the targets present in the other sessions of same the mini-batch
            # ==> (items with the same session are not considered as negatives)
            bs = label_seq_trg.shape[0]
            # build negative mask for each session (bs, #negatives):
            if self.mlm:

                negative_mask = self.compute_neg_mask(label_mlm_mask)
            else:
                negatives = torch.masked_select(label_seq_trg, mask_trg_pad)
                negative_mask = self.compute_neg_mask(mask_trg_pad)
            # If adding extra negative samples: neg_sampling_extra_samples_per_batch > 0
            if self.neg_sampling_extra_samples_per_batch:
                if self.neg_sampling_store_size != 0:
                    if self.neg_sampling_store_pointer == self.neg_sampling_store_rows:
                        # if all examples in the cache were used: re-sample a new cache
                        self.neg_samples = self.generate_neg_samples(
                            length=self.neg_sampling_store_rows
                        )
                        self.neg_sampling_store_pointer = 0
                    # Get a vector of neg_sampling_extra_samples_per_batch for the current batch
                    sample = self.neg_samples[self.neg_sampling_store_pointer].to(
                        self.device
                    )
                    self.neg_sampling_store_pointer += 1
                else:
                    # Sample for each batch without using a cache
                    sample = self.generate_neg_samples(length=1).to(self.device)

                # Concat the additional samples to mini-batch negatives
                negatives = torch.cat([negatives, sample], dim=0)
                # add ones to negative_mask for additional negatives
                negative_mask = torch.cat(
                    [
                        negative_mask,
                        torch.ones((bs, len(sample)), device=self.device).bool(),
                    ],
                    dim=1,
                )
            positives = ~negative_mask
            # flat negative mask : of shape  N_pos_targets x N_negatives
            negative_mask_all = torch.repeat_interleave(
                negative_mask, positives.sum(1), dim=0
            )
            # Get logit scores
            logit_sample = logits_all[:, negatives]
            # Compute loss:
            loss = self.loss_fn(logit_sample, negative_mask_all)

        # Scaling the loss
        loss = loss * self.loss_scale_factor

        if self.rtd and self.training:
            # Add discriminator binary classification task during training
            # Step 1. Generate fake data using generator logits
            if self.rtd_sample_from_batch:
                # sample items from the current batch
                (
                    fake_inputs,
                    discriminator_labels,
                    batch_updates,
                ) = recsys_task.get_fake_data(
                    label_seq,
                    trg_flat,
                    logits_all[:, labels_all],
                    self.rtd_sample_from_batch,
                )
            else:
                # sample items from the whole corpus
                fake_inputs, discriminator_labels, _ = recsys_task.get_fake_data(
                    label_seq,
                    trg_flat,
                    logits_all,
                    self.rtd_sample_from_batch,
                )

            # Step 2. Build interaction embeddings using new replaced itemids
            # TODO: sampling fake side info as well
            if self.rtd_use_batch_interaction:
                # use processed interactions of the current batch
                assert (
                    self.rtd_sample_from_batch
                ), "When rtd_use_batch_interaction, replacement items should be sampled from the current batch, you should set 'rtd_sample_from_batch' to True"
                # detach() is needed to not propagate the discriminator loss through generator
                fake_pos_emb = pos_emb.clone().detach().view(-1, pos_emb.size(2))
                replacement_interaction = fake_pos_emb[batch_updates]
                # replace original masked interactions by fake itemids' interactions
                fake_pos_emb[
                    non_pad_mask.nonzero().flatten(), :
                ] = replacement_interaction
                fake_pos_emb = fake_pos_emb.view(pos_emb.shape)

            else:
                # re-compute interaction embeddings of corrupted sequence of itemids
                inputs[self.label_feature_name] = fake_inputs
                (
                    fake_emb_inp,
                    label_seq,
                    metadata_for_pred_logging,
                ) = self.feature_process(inputs)
                #  Projection layer for corrupted interaction embedding
                if self.rtd_tied_generator:
                    fake_pos_emb = self.merge(fake_emb_inp)

                else:
                    fake_pos_emb = self.merge_disc(fake_emb_inp)

            # Step 3. hidden representation of corrupted input
            if self.rtd_tied_generator:
                # use the generator model for token classification
                fake_pos_emb_pred = self.model(inputs_embeds=fake_pos_emb)[0]
            else:
                # use larger disciminator electra model
                fake_pos_emb_pred = self.discriminator(inputs_embeds=fake_pos_emb)[0]
            # Step 4. get logits for binary pedictions
            fake_pos_emb_pred = self.dense_discriminator(fake_pos_emb_pred)
            fake_pos_emb_pred = self.tf_out_act(fake_pos_emb_pred)
            binary_logits = self.discriminator_prediction(fake_pos_emb_pred).squeeze(-1)
            # Step 5. Get logits for non-padded items
            non_pad_mask = label_seq != self.pad_token
            active_logits = binary_logits.view(-1, fake_pos_emb_pred.shape[1])[
                non_pad_mask
            ]
            active_labels = discriminator_labels[non_pad_mask]
            discriminator_loss = nn.BCEWithLogitsLoss()(
                active_logits, active_labels.float()
            )
            # Step 6. Compute weighted joint training loss
            loss = (discriminator_loss * self.rtd_discriminator_loss_weight) + (
                loss * self.rtd_generator_loss_weight
            )

        outputs = {
            "loss": loss,
            "labels": labels_all,
            "predictions": logits_all,
            "pred_metadata": metadata_for_pred_logging,
            "model_outputs": model_outputs,  # Keep mems, hidden states, attentions if there are in it
        }

        return outputs

    def define_features_layers(self, model_args):
        self.target_dim = None
        self.label_feature_name = None
        self.label_embedding_table_name = None
        self.label_embedding_dim = None

        self.embedding_tables = nn.ModuleDict()
        self.numeric_to_embedding_layers = nn.ModuleDict()
        self.numeric_soft_embeddings = nn.ModuleDict()
        self.features_embedding_projection_to_item_embedding_dim_layers = (
            nn.ModuleDict()
        )

        self.features_layer_norm = nn.ModuleDict()

        self.input_combined_dim = 0

        # set embedding tables
        for cname, cinfo in self.feature_map.items():

            # Ignoring past features to define embedding tables
            if self.session_aware and cname.startswith(
                self.session_aware_features_prefix
            ):
                continue

            if cinfo["dtype"] == "categorical":
                if self.use_ohe_item_ids_inputs:
                    feature_size = cinfo["cardinality"]
                else:
                    if "is_label" in cinfo and cinfo["is_label"]:
                        if model_args.item_embedding_dim is not None:
                            embedding_size = model_args.item_embedding_dim
                        # This condition is just to keep compatibility with the experiments of SIGIR paper
                        # (where item embedding dim was always equal to d_model)
                        elif model_args.mf_constrained_embeddings:
                            embedding_size = model_args.d_model
                        else:
                            embedding_size = get_embedding_size_from_cardinality(
                                cinfo["cardinality"],
                                multiplier=self.embedding_dim_from_cardinality_multiplier,
                            )
                        feature_size = embedding_size

                        self.item_embedding_dim = embedding_size

                        self.label_feature_name = cname
                        self.label_embedding_table_name = cinfo["emb_table"]
                        self.label_embedding_dim = embedding_size

                    else:
                        if self.features_same_size_item_embedding:
                            if self.label_embedding_dim:
                                embedding_size = self.label_embedding_dim
                                feature_size = embedding_size
                            else:
                                raise ValueError(
                                    "Make sure that the item id (label feature) is the first in the YAML features config file."
                                )
                        else:
                            embedding_size = get_embedding_size_from_cardinality(
                                cinfo["cardinality"],
                                multiplier=self.embedding_dim_from_cardinality_multiplier,
                            )
                            feature_size = embedding_size

                            if (
                                self.input_features_aggregation
                                == "elementwise_sum_multiply_item_embedding"
                            ):
                                self.features_embedding_projection_to_item_embedding_dim_layers[
                                    cname
                                ] = nn.Linear(
                                    embedding_size,
                                    self.label_embedding_dim,
                                    bias=True,
                                )
                                feature_size = self.label_embedding_dim

                    self.embedding_tables[cinfo["emb_table"]] = nn.Embedding(
                        cinfo["cardinality"],
                        embedding_size,
                        padding_idx=self.pad_token,
                    ).to(self.device)

                    # Added to initialize embeddings
                    if "is_label" in cinfo and cinfo["is_label"]:
                        embedding_init_std = self.item_id_embeddings_init_std
                    else:
                        embedding_init_std = self.other_embeddings_init_std

                    with torch.no_grad():
                        self.embedding_tables[cinfo["emb_table"]].weight.normal_(
                            0.0, embedding_init_std
                        )

                logger.info(
                    "Categ Feature: {} - Cardinality: {} - Feature Size: {}".format(
                        cname, cinfo["cardinality"], feature_size
                    )
                )

            elif cinfo["dtype"] in ["long", "float"]:

                if self.numeric_features_project_to_embedding_dim > 0:
                    if self.features_same_size_item_embedding:
                        if self.label_embedding_dim:
                            feature_size = self.label_embedding_dim
                        else:
                            raise ValueError(
                                "Make sure that the item id (label feature) is the first in the YAML features config file."
                            )
                    else:
                        feature_size = self.numeric_features_project_to_embedding_dim

                    if self.numeric_features_soft_one_hot_encoding_num_embeddings > 0:
                        self.numeric_soft_embeddings[cname] = SoftEmbedding(
                            num_embeddings=self.numeric_features_soft_one_hot_encoding_num_embeddings,
                            embeddings_dim=feature_size,
                            embeddings_init_std=self.other_embeddings_init_std,
                        )

                        if (
                            self.input_features_aggregation
                            == "elementwise_sum_multiply_item_embedding"
                        ):
                            self.features_embedding_projection_to_item_embedding_dim_layers[
                                cname
                            ] = nn.Linear(
                                feature_size,
                                self.label_embedding_dim,
                                bias=True,
                            )
                            feature_size = self.label_embedding_dim
                    else:
                        if (
                            self.input_features_aggregation
                            == "elementwise_sum_multiply_item_embedding"
                        ):
                            project_scalar_to_embedding_dim = self.label_embedding_dim
                        else:
                            project_scalar_to_embedding_dim = (
                                self.numeric_features_project_to_embedding_dim
                            )
                        feature_size = project_scalar_to_embedding_dim

                        self.numeric_to_embedding_layers[cname] = nn.Linear(
                            1, project_scalar_to_embedding_dim, bias=True
                        )

                else:
                    feature_size = 1

                logger.info(
                    "Numerical Feature: {} - Feature Size: {}".format(
                        cname, feature_size
                    )
                )

            elif cinfo["is_control"]:
                # Control features are not used as input for the model
                continue
            else:
                raise NotImplementedError

            self.input_combined_dim += feature_size

            if self.layer_norm_featurewise:
                self.features_layer_norm[cname] = nn.LayerNorm(
                    normalized_shape=feature_size
                )

            if "is_label" in cinfo and cinfo["is_label"]:
                self.target_dim = cinfo["cardinality"]

        if self.input_features_aggregation == "elementwise_sum_multiply_item_embedding":
            self.input_combined_dim = self.item_embedding_dim

        if self.target_dim == None:
            raise RuntimeError("label column is not declared in feature map.")

    def feature_process(self, inputs):

        label_seq, output = None, []
        metadata_for_pred_logging = {}

        transformed_features = OrderedDict()
        for cname, cinfo in self.feature_map.items():

            cdata = inputs[cname]

            if "is_label" in cinfo and cinfo["is_label"]:
                label_seq = cdata

            if cinfo["dtype"] == "categorical":
                cdata = cdata.long()

                # Applies Stochastic Shared Embeddings if training
                if (
                    self.stochastic_shared_embeddings_replacement_prob > 0.0
                    and not self.use_ohe_item_ids_inputs
                    and self.training
                ):
                    with torch.no_grad():
                        cdata_non_zero_mask = cdata != self.pad_token

                        sse_prob_replacement_matrix = torch.full(
                            cdata.shape,
                            self.stochastic_shared_embeddings_replacement_prob,
                            device=self.device,
                        )
                        sse_replacement_mask = (
                            torch.bernoulli(sse_prob_replacement_matrix).bool()
                            & cdata_non_zero_mask
                        )
                        n_values_to_replace = sse_replacement_mask.sum()

                        cdata_flattened_non_zero = torch.masked_select(
                            cdata, cdata_non_zero_mask
                        )

                        sampled_values_to_replace = cdata_flattened_non_zero[
                            torch.randperm(cdata_flattened_non_zero.shape[0])
                        ][:n_values_to_replace]

                        cdata[sse_replacement_mask] = sampled_values_to_replace

                if "is_label" in cinfo and cinfo["is_label"]:
                    if self.use_ohe_item_ids_inputs:
                        cdata = torch.nn.functional.one_hot(
                            cdata, num_classes=self.target_dim
                        ).float()
                    else:
                        cdata = self.embedding_tables[cinfo["emb_table"]](cdata)
                else:
                    cdata = self.embedding_tables[cinfo["emb_table"]](cdata)

                    if (
                        self.input_features_aggregation
                        == "elementwise_sum_multiply_item_embedding"
                        and not self.features_same_size_item_embedding
                    ):
                        cdata = self.features_embedding_projection_to_item_embedding_dim_layers[
                            cname
                        ](
                            cdata
                        )

            elif cinfo["dtype"] in ["long", "float"]:
                if cinfo["dtype"] == "long":
                    cdata = cdata.unsqueeze(-1).long()
                elif cinfo["dtype"] == "float":
                    cdata = cdata.unsqueeze(-1).float()

                if self.numeric_features_project_to_embedding_dim > 0:
                    if self.numeric_features_soft_one_hot_encoding_num_embeddings > 0:
                        cdata = self.numeric_soft_embeddings[cname](cdata)

                        if (
                            self.input_features_aggregation
                            == "elementwise_sum_multiply_item_embedding"
                            and not self.features_same_size_item_embedding
                        ):
                            cdata = self.features_embedding_projection_to_item_embedding_dim_layers[
                                cname
                            ](
                                cdata
                            )
                    else:
                        cdata = self.numeric_to_embedding_layers[cname](cdata)

            elif cinfo["is_control"]:
                # Control features are not used as input for the model
                continue
            else:
                raise NotImplementedError

            # Applying layer norm for each feature
            if self.layer_norm_featurewise:
                cdata = self.features_layer_norm[cname](cdata)

            transformed_features[cname] = cdata

            # Keeping item metadata features that will
            if (
                "log_with_preds_as_metadata" in cinfo
                and cinfo["log_with_preds_as_metadata"] == True
            ):
                metadata_for_pred_logging[cname] = inputs[cname].detach()

        if self.session_aware:
            # Concatenates past sessions before the session with the current session, for each feature
            # assuming that features from past sessions have a common prefix
            features_to_delete = []
            for fname in transformed_features.keys():
                if not fname.startswith(self.session_aware_features_prefix):
                    past_fname = self.session_aware_features_prefix + fname
                    if past_fname in transformed_features:
                        transformed_features[fname] = torch.cat(
                            [
                                transformed_features[past_fname],
                                transformed_features[fname],
                            ],
                            axis=1,
                        )
                    features_to_delete.append(past_fname)
            for past_fname in features_to_delete:
                del transformed_features[past_fname]

        if len(transformed_features) > 1:
            if self.input_features_aggregation == "concat":
                output = torch.cat(list(transformed_features.values()), dim=-1)
            elif (
                self.input_features_aggregation
                == "elementwise_sum_multiply_item_embedding"
            ):
                additional_features_sum = torch.zeros_like(
                    transformed_features[self.label_feature_name], device=self.device
                )
                for k, v in transformed_features.items():
                    if k != self.label_feature_name:
                        if (self.feature_map[k]["dtype"] == "categorical") or (
                            self.feature_map[k]["dtype"] in ["long", "float"]
                            and self.numeric_features_project_to_embedding_dim > 0
                        ):
                            additional_features_sum += v

                item_id_embedding = transformed_features[self.label_feature_name]

                output = item_id_embedding * (additional_features_sum + 1.0)
            else:
                raise ValueError("Invalid value for --input_features_aggregation.")
        else:
            output = list(transformed_features.values())[0]

        return output, label_seq, metadata_for_pred_logging

    def remove_pad_3d(self, inp_tensor, non_pad_mask):
        # inp_tensor: (n_batch x seqlen x emb_dim)
        inp_tensor = inp_tensor.flatten(end_dim=1)
        inp_tensor_fl = torch.masked_select(
            inp_tensor, non_pad_mask.unsqueeze(1).expand_as(inp_tensor)
        )
        out_tensor = inp_tensor_fl.view(-1, inp_tensor.size(1))
        return out_tensor

    def remove_pad_4d(self, inp_tensor, non_pad_mask):
        # inp_tensor: (n_batch x seqlen x n_negex x emb_dim)
        inp_tensor_fl = inp_tensor.reshape(-1, inp_tensor.size(2), inp_tensor.size(3))
        inp_tensor_fl = torch.masked_select(
            inp_tensor_fl,
            non_pad_mask.unsqueeze(1).unsqueeze(2).expand_as(inp_tensor_fl),
        )
        out_tensor = inp_tensor_fl.view(-1, inp_tensor.size(2), inp_tensor.size(3))
        return out_tensor

    def set_items_freq_for_sampling(self, items_sorted_freq_series):
        self.items_ids_sorted_by_freq = torch.Tensor(
            items_sorted_freq_series.index.values, device=self.device
        )
        self.items_freq_sorted = torch.Tensor(
            items_sorted_freq_series.values, device=self.device
        )

        # If should adding extra negative samples to the batch ones
        if (
            self.neg_sampling_store_size != 0
        ) and self.neg_sampling_extra_samples_per_batch > 0:
            # Generate a cumulative distribution of frequency (sorted in ascending order), so that more popular items can be sampled more often
            self.items_freq_sorted_norm = (
                self.items_freq_sorted ** self.neg_sampling_alpha
            )
            self.items_freq_sorted_norm = self.items_freq_sorted_norm.cumsum(
                dim=0
            ) / self.items_freq_sorted_norm.sum(dim=0)
            self.items_freq_sorted_norm[-1] = 1

            # Defines a cache that pre-stores N="neg_sampling_store_size" negative samples
            self.neg_sampling_store_rows = (
                self.neg_sampling_store_size
                // self.neg_sampling_extra_samples_per_batch
            )
            if self.neg_sampling_store_rows <= 1:
                self.neg_sampling_store_rows = 0
                print("No negative samples store was used.")
            else:
                self.neg_samples = self.generate_neg_samples(
                    length=self.neg_sampling_store_rows
                )
                self.neg_sampling_store_pointer = 0
                print(
                    "Created sample store with {} batches of samples (type=CPU)".format(
                        self.neg_sampling_store_rows
                    )
                )
        else:
            print("No example store was used")

    def generate_neg_samples(self, length):
        """
        Args:
            length: the number of vectors of shape self.neg_sampling_extra_samples_per_batch to store in cache memory
        return:
            sample: Tensor of negative samples of shape length x self.neg_sampling_extra_samples_per_batch
        """

        if self.neg_sampling_alpha:
            samples_idx = torch.searchsorted(
                self.items_freq_sorted_norm,
                torch.rand(
                    self.neg_sampling_extra_samples_per_batch * length,
                    device=self.device,
                ),
            )
            # Retrieves the correct item ids from the sampled indices over the cumulative prob distribution
            sampled_item_ids = self.items_ids_sorted_by_freq[samples_idx]
        else:
            n_items = self.items_freq_sorted_norm.shape[0]
            sampled_item_ids = torch.randint(
                0,
                n_items,
                size=(self.neg_sampling_extra_samples_per_batch * length,),
                device=self.device,
            )
        if length > 1:
            sampled_item_ids = sampled_item_ids.reshape(
                (length, self.neg_sampling_extra_samples_per_batch)
            )
        return sampled_item_ids

    def compute_neg_mask(self, positive_mask):
        """
        Args:
            positive_mask: Tensor of shape bs x seq_len: mask  input where target is on padding token
        Return:
            negative_mask: Tensor of shape #pos_targets x negatives to specify the negative items
                            for each positive target
        """
        # TODO: Refactor the code to not use  For loop
        bs, _ = positive_mask.shape
        N_neg = positive_mask.flatten().sum()
        pos_target_per_session = positive_mask.sum(1)
        pos_target_per_session = torch.cat(
            [torch.Tensor([0], device=self.device), pos_target_per_session]
        )
        cumul_pos_target = pos_target_per_session.cumsum(dim=0)
        # define mask over all mini-batch negatives
        mask = torch.zeros(bs, N_neg, device=self.device)
        for i in range(bs):
            mask[i, cumul_pos_target[i] : cumul_pos_target[i + 1]] = 1
        return ~mask.bool()


def nll_1d(items_prob, _label=None):
    # https://github.com/gabrielspmoreira/chameleon_recsys/blob/da7f73a2b31d6867d444eded084044304b437413/nar_module/nar/nar_model.py#L639
    items_prob = torch.exp(items_prob)
    positive_prob = items_prob[:, 0]
    xe_loss = torch.log(positive_prob)
    cosine_sim_loss = -torch.mean(xe_loss)
    return cosine_sim_loss


class SoftEmbedding(nn.Module):
    """
    Soft-one hot encoding embedding, from https://arxiv.org/pdf/1708.00065.pdf
    """

    def __init__(self, num_embeddings, embeddings_dim, embeddings_init_std=0.05):
        super().__init__()
        self.embedding_table = nn.Embedding(num_embeddings, embeddings_dim)
        with torch.no_grad():
            self.embedding_table.weight.normal_(0.0, embeddings_init_std)
        self.projection_layer = nn.Linear(1, num_embeddings, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_numeric):
        weights = self.softmax(self.projection_layer(input_numeric))
        soft_one_hot_embeddings = (
            weights.unsqueeze(-1) * self.embedding_table.weight
        ).sum(-2)
        return soft_one_hot_embeddings


# From https://github.com/NingAnMe/Label-Smoothing-for-CrossEntropyLoss-PyTorch
class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(
            targets, inputs.size(-1), self.smoothing
        )
        # The following line was commented because the inpus were already processed by log_softmax()
        # lsm = F.log_softmax(inputs, -1)
        lsm = inputs

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss
