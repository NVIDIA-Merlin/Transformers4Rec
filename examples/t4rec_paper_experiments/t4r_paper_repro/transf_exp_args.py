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
from dataclasses import dataclass, field
from typing import Optional

from transformers4rec.torch import T4RecTrainingArguments


@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to dataset."},
    )

    features_schema_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path with the dataset schema in protobuf text format"},
    )

    start_time_window_index: Optional[int] = field(
        default=1,
        metadata={
            "help": "The start time window index to use for training & evaluation."
            "The framework expects data to be split in contiguous non-overlapping time "
            "windows, with fixed time units (e.g. days, hours)."
        },
    )

    final_time_window_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "The final time window index to use for training & evaluation."
            "The framework expects data to be split in contiguous non-overlapping time "
            "windows, with fixed time units (e.g. days, hours)."
        },
    )

    time_window_folder_pad_digits: Optional[int] = field(
        default=4,
        metadata={
            "help": "The framework expects data to be split in contiguous non-overlapping "
            "time windows, with fixed time units (e.g. days, hours)."
            "Each time window is represented as a folder within --data_path, named with "
            "padded zeros on the left, according to the number"
            " of digits defined in --time_window_folder_digits. For example, for "
            "--time_window_folder_digits 4, folders should be: '0001', '0002', ..."
        },
    )

    no_incremental_training: bool = field(
        default=False,
        metadata={
            "help": "Indicates whether the model should be trained incrementally over "
            "time (False) or trained using only the time window (True) defined "
            "in --training_time_window_size"
        },
    )

    training_time_window_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "Window size of time units (e.g. days or hours) to use for training, "
            "when not using --incremental_training. "
            "If --training_time_window_size 0, the training window will start on index 1 and "
            "finish right before the current eval index (incremental evaluation)"
            "The framework expects data to be split in contiguous non-overlapping time windows, "
            "with fixed time units (e.g. days, hours)."
        },
    )

    use_side_information_features: bool = field(
        default=False,
        metadata={
            "help": "By default (False) only the feature tagged in the schema as 'item_id' "
            "is used by the model. "
            "If enabled, uses all the available features in the model"
        },
    )


@dataclass
class ModelArguments:
    input_features_aggregation: Optional[str] = field(
        default="concat",
        metadata={
            "help": "How input features are merged. Supported options: concat |"
            " elementwise_sum_multiply_item_embedding"
        },
    )

    model_type: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "Type of the sequential model. Can be: "
            "gpt2|transfoxl|xlnet|reformer|longformer"
        },
    )

    tf_out_activation: Optional[str] = field(
        default="tanh",
        metadata={"help": "transformer output activation: 'tanh' OR 'relu'"},
    )

    # args for MLM task

    mlm: bool = field(
        default=False,
        metadata={"help": "Use Masked Language Modeling (Cloze objective) for training."},
    )

    mlm_probability: Optional[float] = field(
        default=0.15,
        metadata={
            "help": "Ratio of tokens to mask (set target) from an original sequence. "
            "There is a hard constraint that ensures that each sequence have at "
            "least one target (masked) and one non-masked item, for effective learning. "
            "Thus, if the sequence has more than 2 items, this is the probability of "
            "the additional items to be masked"
        },
    )

    # args for PLM task

    plm: bool = field(
        default=False,
        metadata={"help": "Use Permutation Language Modeling (XLNET objective) for training."},
    )

    plm_probability: Optional[float] = field(
        default=0.25,
        metadata={
            "help": "Ratio of tokens to unmask to form the surrounding context of the masked span"
        },
    )
    plm_max_span_length: Optional[int] = field(
        default=5,
        metadata={"help": "the maximum length of segment to mask for partial prediction"},
    )

    plm_mask_input: Optional[bool] = field(
        default=False, metadata={"help": "Mask input of XLNET as in AE models or not"}
    )

    plm_permute_all: Optional[bool] = field(
        default=False, metadata={"help": "Permute all non padded items"}
    )

    # args for RTD task
    rtd: bool = field(
        default=False,
        metadata={"help": "Use Replaced Token Detection (ELECTRA objective) for training."},
    )

    rtd_sample_from_batch: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Sample replacement itemids from the whole corpus (False) or only from "
            "the current batch (True)"
        },
    )

    rtd_use_batch_interaction: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use batch processed item interactions for building the corrupted sequence "
            "of itemids to feed to the discriminator. "
            "This option is only available when rtd_sample_from_batch is set to True "
        },
    )

    rtd_discriminator_loss_weight: Optional[float] = field(
        default=50, metadata={"help": "Weight coefficient for the discriminator loss"}
    )

    rtd_generator_loss_weight: Optional[float] = field(
        default=1, metadata={"help": "Weight coefficient for the generator loss"}
    )

    rtd_tied_generator: Optional[bool] = field(
        default=False, metadata={"help": "Tie all generator/discriminator weights?"}
    )

    """
    # Only when --rtd_tied_generator False
    electra_generator_hidden_size: Optional[float] = field(
        default=0.25,
        metadata={
            "help": "Frac. of discriminator hidden size for smaller generator (only for "
                    "--model_type electra)"
        },
    )
    """

    # args for Transformers or RNNs

    d_model: Optional[int] = field(
        default=256,
        metadata={"help": "size of hidden states (or internal states) for RNNs and Transformers"},
    )
    n_layer: Optional[int] = field(
        default=12, metadata={"help": "number of layers for RNNs and Transformers"}
    )
    n_head: Optional[int] = field(
        default=4, metadata={"help": "number of attention heads for Transformers"}
    )
    layer_norm_eps: Optional[float] = field(
        default=1e-12,
        metadata={"help": "The epsilon used by the layer normalization layers for Transformers"},
    )
    initializer_range: Optional[float] = field(
        default=0.02,
        metadata={
            "help": "The standard deviation of the truncated_normal_initializer for "
            "initializing all weight matrices for Transformers"
        },
    )
    hidden_act: Optional[str] = field(
        default="gelu",
        metadata={
            "help": "The non-linear activation function (function or string) in Transformers. "
            "'gelu', 'relu' and 'swish' are supported."
        },
    )
    dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, "
            "encoder, and decoders for Transformers and RNNs"
        },
    )

    # args for XLNET
    summary_type: Optional[str] = field(
        default="last",
        metadata={
            "help": "How to summarize the vector representation of the sequence'last', 'first', "
            "'mean', 'attn' are supported"
        },
    )

    # args for ALBERT

    num_hidden_groups: Optional[int] = field(
        default=1,
        metadata={
            "help": "(ALBERT) Number of groups for the hidden layers, parameters in the same "
            "group are shared."
        },
    )
    inner_group_num: Optional[int] = field(
        default=1,
        metadata={"help": "(ALBERT) The number of inner repetition of attention and ffn."},
    )

    # General training args
    eval_on_last_item_seq_only: bool = field(
        default=False,
        metadata={
            "help": "Evaluate metrics only on predictions for the last item of the sequence "
            "(rather then evaluation for all next-item predictions)."
        },
    )
    train_on_last_item_seq_only: bool = field(
        default=False,
        metadata={
            "help": "Train only for predicting the last item of the sequence (rather then "
            "training to predict for all next-item predictions) (only for Causal LM)."
        },
    )

    # Weight-tying (tying embeddings)
    mf_constrained_embeddings: bool = field(
        default=False,
        metadata={
            "help": "Implements the tying embeddings technique, in which the item id embedding "
            "table weights are shared with the last network layer which predicts over all items"
            "This is equivalent of performing a matrix factorization (dot product multiplication) "
            "operation between the Transformers output and the item embeddings."
            "This option requires the item id embeddings to have the same dimensions of the last "
            "network layer."
        },
    )

    # Input features representation
    item_embedding_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Dimension of the item embedding. If it is None, a heuristic method used "
            "to define the dimension based on items cardinality. "
            "If --mf_constrained_embeddings or --constrained_embeddings are enabled, "
            "the output of transformers (dimension defined by --d_model) will "
            "be projected to the same dimension as the item embedding (tying embedding), "
            "just before the output layer. "
            "You can define the item embedding dim using --item_embedding_dim or let the "
            "size to be defined automatically based on its cardinality multiplied by the "
            "--embedding_dim_from_cardinality_multiplier factor."
        },
    )

    numeric_features_project_to_embedding_dim: Optional[int] = field(
        default=0,
        metadata={
            "help": "Uses a fully-connected layet to project a numeric scalar feature to an "
            "embedding with this dimension. If --features_same_size_item_embedding, "
            "the embedding will have the same size as the item embedding"
        },
    )

    numeric_features_soft_one_hot_encoding_num_embeddings: Optional[int] = field(
        default=0,
        metadata={
            "help": "If greater than zero, enables soft one-hot encoding technique for "
            "numerical features (https://arxiv.org/pdf/1708.00065.pdf). It will be "
            "created an embedding table with dimension defined by "
            "(--numeric_features_soft_one_hot_encoding_num_embeddings, "
            "--numeric_features_project_to_embedding_dim)"
        },
    )

    stochastic_shared_embeddings_replacement_prob: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Probability of the embedding of a categorical feature be replaced by "
            "another an embedding of the same batch"
        },
    )

    softmax_temperature: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Softmax temperature, used to reduce model overconfidence, so that "
            "softmax(logits / T). Value 1.0 reduces to regular softmax."
        },
    )

    label_smoothing: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Applies label smoothing using as alpha this parameter value. "
            "It helps overconfidence of models and calibration of the predictions."
        },
    )

    embedding_dim_from_cardinality_multiplier: Optional[float] = field(
        default=2.0,
        metadata={
            "help": "Used to define the feature embedding dim based on its cardinality. "
            "The formula is embedding_size = int(math.ceil(math.pow(cardinality, 0.25) "
            "* multiplier))"
        },
    )

    item_id_embeddings_init_std: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Uniform distribution maximum and minimum (-bound) value to be used to "
            "initialize the item id embedding (usually must be higher than "
            "--categs_embeddings_init_uniform_bound, as those weights are also used "
            "as the output layer when --mf_constrained_embeddings)"
        },
    )

    other_embeddings_init_std: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Uniform distribution maximum and minimum (-bound) value to be used "
            "to initialize the other feature embeddings (other than the item_id, "
            "which is defined by --item_id_embeddings_init_uniform_bound)"
        },
    )

    layer_norm_featurewise: bool = field(
        default=False,
        metadata={
            "help": "Enables layer norm for each feature individually, before their aggregation."
        },
    )

    attn_type: str = field(
        default="uni",
        metadata={"help": "The type of attention. Use 'uni' for Causal LM and 'bi' for Masked LM"},
    )

    # TODO: Args to be used in the script
    input_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "The dropout probability of the input embeddings, before being combined with "
            "feed-forward layers"
        },
    )

    # The following arguments had a single value set for all experiments and are here
    # just to accept the same arguments, but are not really used
    loss_type: Optional[str] = field(
        default="cross_entropy",
        metadata={"help": "Type of Loss function: cross_entropy|top1|top1_max|bpr|bpr_max_reg"},
    )

    similarity_type: Optional[str] = field(
        default="concat_mlp",
        metadata={"help": "how to compute similarity of sequences"},
    )

    inp_merge: Optional[str] = field(
        default="mlp", metadata={"help": "input merge mechanism: 'mlp' OR 'attn'"}
    )

    learning_rate_warmup_steps: int = field(
        default=0,
        metadata={
            "help": "Number of steps to linearly increase the learning rate from 0 to "
            "the specified initial learning rate schedule. Valid for "
            "--learning_rate_schedule = constant_with_warmup | linear_with_warmup | "
            "cosine_with_warmup"
        },
    )

    avg_session_length: int = field(
        default=None,
        metadata={
            "help": "When --eval_on_last_item_seq_only False, this conservative estimate of "
            "the avg. session length (rounded up to the next int) "
            "is used to estimate the number of interactions from the batch_size (# sessions) "
            "so that the tensor that accumulates all predictions is sufficient to concatenate "
            "all predictions"
        },
    )

    # Sampled Softmax
    sampled_softmax: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables sampled softmax for training."},
    )

    sampled_softmax_max_n_samples: Optional[int] = field(
        default=1000,
        metadata={
            "help": "Max number of unique negative samples for sampled softmax."
            "The number of samples might differ across batches, as the exact number "
            "of unique items might not be the same."
        },
    )


@dataclass
class TrainingArguments(T4RecTrainingArguments):
    ##################################################################################
    # Arguments kept only for compatibility with the original Transformers4Rec paper
    # reproducibility script

    # This argument will be copied to the default Trainer max_sequence_length arg
    session_seq_length_max: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum length of a session (for padding and trimming). "
            "For sequential recommendation, this is the maximum length of the sequence"
        },
    )

    # This argument will be copied to the default Trainer lr_scheduler_type arg
    learning_rate_schedule: str = field(
        default=None,
        metadata={
            "help": "Learning Rate schedule (restarted for each training day). "
            "Valid values: constant_with_warmup | linear_with_warmup | cosine_with_warmup"
        },
    )

    # This argument is not used, just for command line compatibility
    validate_every: int = field(
        default=-1,
        metadata={
            "help": "Run validation set every this epoch. "
            "-1 means no validation is used (default: -1)"
        },
    )
