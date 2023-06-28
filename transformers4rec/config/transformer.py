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

import transformers
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.models.utils.registry import Registry

transformer_registry: Registry = Registry("transformers")


TRANSFORMER_CONFIG_PARAMETER_DOCSTRING = """        
        d_model: int
            The  hidden dimension of the transformer layer.
        n_head: int
            The number of attention heads in each transformer layer.
        n_layer: int
            The number of transformer layers to stack.
        total_seq_length: int
            The maximum sequence length.
        hidden_act: str, optional
            The activation function in the hidden layers.
            By default 'gelu'
        initializer_range: float, optional
            The standard deviation of the `truncated_normal_initializer`
            for initializing all transformer's weights parameters.
            By default 0.01
        layer_norm_eps: float, optional
            The epsilon used by the layer normalization layers.
            By default 0.03
        dropout: float, optional
            The dropout probability. By default 0.3
        pad_token: int, optional
            The padding token ID. By default 0
        log_attention_weights: bool, optional
            Whether to log attention weights. By default False
"""


class T4RecConfig:
    """A class responsible for setting the configuration of the transformers class
    from Hugging Face and returning the corresponding T4Rec model.
    """

    def to_huggingface_torch_model(self):
        """
        Instantiate a Hugging Face transformer model based on
        the configuration parameters of the class.

        Returns
        -------
        transformers.PreTrainedModel
            The Hugging Face transformer model.
        """
        model_cls = transformers.MODEL_MAPPING[self.transformers_config_cls]

        return model_cls(self)

    def to_torch_model(
        self,
        input_features,
        *prediction_task,
        task_blocks=None,
        task_weights=None,
        loss_reduction="mean",
        **kwargs
    ):
        """Links the Hugging Face transformer model to the given input block and prediction tasks,
        and returns a T4Rec model.

        Parameters
        ----------
        input_features: torch4rec.TabularSequenceFeatures
            The sequential block that represents the input features and
            defines the masking strategy for training and evaluation.
        prediction_task: torch4rec.PredictionTask
            One or multiple prediction tasks.
        task_blocks: list, optional
            List of task-specific blocks that we apply on top of the HF transformer's output.
        task_weights: list, optional
            List of the weights to use for combining the tasks losses.
        loss_reduction: str, optional
            The reduction to apply to the prediction losses, possible values are:
                'none': no reduction will be applied,
                'mean': the weighted mean of the output is taken,
                'sum': the output will be summed.
            By default: 'mean'.

        Returns
        -------
        torch4rec.Model
            The T4Rec torch model.

        Raises
        ------
        ValueError
            If input block or prediction task is of the wrong type.
        """
        from .. import torch as torch4rec

        if not isinstance(input_features, torch4rec.TabularSequenceFeatures):
            raise ValueError("`input_features` must an instance of SequentialTabularFeatures")
        if not all(isinstance(t, torch4rec.PredictionTask) for t in prediction_task):
            raise ValueError(
                "`task` is of the wrong type, please provide one or multiple "
                "instance(s) of PredictionTask"
            )

        body = torch4rec.SequentialBlock(
            input_features, torch4rec.TransformerBlock(self, masking=input_features.masking)
        )

        return torch4rec.Head(
            body,
            *prediction_task,
            task_blocks=task_blocks,
            task_weights=task_weights,
            loss_reduction=loss_reduction,
        ).to_model(**kwargs)

    @property
    def transformers_config_cls(self):
        return self.__class__.__bases__[1]

    @classmethod
    def build(cls, *args, **kwargs):
        raise NotImplementedError


@transformer_registry.register("reformer")
class ReformerConfig(T4RecConfig, transformers.ReformerConfig):
    """Subclass of T4RecConfig and transformers.ReformerConfig from Hugging Face.
    It handles configuration for Reformer layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        axial_pos_shape_first_dim=4,
        **kwargs
    ):
        """
        Creates an instance of ReformerConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}
        axial_pos_shape_first_dim: int, optional
            The first dimension of the axial position encodings.
            During training, the product of the position dims has to be equal to the sequence length.

        Returns
        -------
        ReformerConfig
            An instance of ReformerConfig.
        """
        # To account for target positions at inference mode, we extend the maximum sequence length.
        total_seq_length = total_seq_length + 2
        return cls(
            hidden_size=d_model,
            attention_head_size=d_model,
            attn_layers=["local", "lsh"] * (n_layer // 2) if n_layer > 2 else ["local"],
            num_hidden_layers=n_layer,
            feed_forward_size=d_model * 4,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=dropout,
            lsh_attention_probs_dropout_prob=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            max_position_embeddings=total_seq_length,
            axial_pos_embds_dim=[
                d_model // 2,
                d_model // 2,
            ],
            axial_pos_shape=[
                axial_pos_shape_first_dim,
                total_seq_length // axial_pos_shape_first_dim,
            ],
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("gtp2")
@docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
class GPT2Config(T4RecConfig, transformers.GPT2Config):
    """Subclass of T4RecConfig and transformers.GPT2Config from Hugging Face.
    It handles configuration for GPT2 layers in the context of T4Rec models.
    """

    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of GPT2Config with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        GPT2Config
            An instance of GPT2Config.
        """
        return cls(
            n_embd=d_model,
            n_inner=d_model * 4,
            n_layer=n_layer,
            n_head=n_head,
            activation_function=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            n_positions=total_seq_length,
            n_ctx=total_seq_length,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("longformer")
class LongformerConfig(T4RecConfig, transformers.LongformerConfig):
    """Subclass of T4RecConfig and transformers.LongformerConfig from Hugging Face.
    It handles configuration for LongformerConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of LongformerConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        LongformerConfig
            An instance of LongformerConfig.
        """
        # To account for target positions at inference mode, we extend the maximum sequence length.
        total_seq_length = total_seq_length + 2
        return cls(
            hidden_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            attention_window=total_seq_length,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("electra")
class ElectraConfig(T4RecConfig, transformers.ElectraConfig):
    """Subclass of T4RecConfig and transformers.ElectraConfig from Hugging Face.
    It handles configuration for ElectraConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of ElectraConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        ElectraConfig
            An instance of ElectraConfig.
        """
        # To account for target positions at inference mode, we extend the maximum sequence length.
        total_seq_length = total_seq_length + 2
        return cls(
            hidden_size=d_model,
            embedding_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            intermediate_size=d_model * 4,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=dropout,
            max_position_embeddings=total_seq_length,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("albert")
class AlbertConfig(T4RecConfig, transformers.AlbertConfig):
    """Subclass of T4RecConfig and transformers.AlbertConfig from Hugging Face.
    It handles configuration for AlbertConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of AlbertConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        AlbertConfig
            An instance of AlbertConfig.
        """
        # To account for target positions at inference mode, we extend the maximum sequence length.
        total_seq_length = total_seq_length + 2
        return cls(
            hidden_size=d_model,
            num_attention_heads=n_head,
            num_hidden_layers=n_layer,
            hidden_act=hidden_act,
            intermediate_size=d_model * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=total_seq_length,
            embedding_size=d_model,  # should be same as dimension of the input to ALBERT
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("xlnet")
@docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
class XLNetConfig(T4RecConfig, transformers.XLNetConfig):
    """Subclass of T4RecConfig and transformers.XLNetConfig from Hugging Face.
    It handles configuration for XLNetConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length=None,
        attn_type="bi",
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        mem_len=1,
        **kwargs
    ):
        """
        Creates an instance of XLNetConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}
        mem_len: int,
            The number of tokens to be cached. Pre-computed key/value pairs
            from a previous forward pass are stored and won't be re-computed.
            This parameter is especially useful for long sequence modeling where
            different batches may truncate the entire sequence.
            Tasks like user-aware recommendation could benefit from this feature.
            By default, this parameter is set to 1, which means no caching is used.

        Returns
        -------
        XLNetConfig
            An instance of XLNetConfig.
        """
        return cls(
            d_model=d_model,
            d_inner=d_model * 4,
            n_layer=n_layer,
            n_head=n_head,
            attn_type=attn_type,
            ff_activation=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,
            mem_len=mem_len,
            **kwargs,
        )


@transformer_registry.register("bert")
class BertConfig(T4RecConfig, transformers.BertConfig):
    """Subclass of T4RecConfig and transformers.BertConfig from Hugging Face.
    It handles configuration for BertConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of BertConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        BertConfig
            An instance of BertConfig.
        """
        # To account for target positions at inference mode, we extend the maximum sequence length.
        total_seq_length = total_seq_length + 2
        return cls(
            hidden_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            max_position_embeddings=total_seq_length,
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("roberta")
class RobertaConfig(T4RecConfig, transformers.RobertaConfig):
    """Subclass of T4RecConfig and transformers.RobertaConfig from Hugging Face.
    It handles configuration for RobertaConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of RobertaConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        RobertaConfig
            An instance of RobertaConfig.
        """
        # To account for target positions at inference mode, we extend the maximum sequence length.
        total_seq_length = total_seq_length + 2
        return cls(
            hidden_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            max_position_embeddings=total_seq_length,
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("transfo-xl")
class TransfoXLConfig(T4RecConfig, transformers.TransfoXLConfig):
    """Subclass of T4RecConfig and transformers. TransfoXLConfig from Hugging Face.
    It handles configuration for TransfoXLConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of TransfoXLConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        TransfoXLConfig
            An instance of TransfoXLConfig.
        """
        return cls(
            d_model=d_model,
            d_embed=d_model,
            n_layer=n_layer,
            n_head=n_head,
            d_inner=d_model * 4,
            hidden_act=hidden_act,
            untie_r=True,
            attn_type=0,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,  # As the input_embeds will be fed in the forward function, limits the memory reserved by the internal input embedding table, which will not be used
            mem_len=1,  # We do not use mems, because we feed the full sequence to the Transformer models and not sliding segments (which is useful for the long sequences in NLP. As setting mem_len to 0 leads to NaN in loss, we set it to one, to minimize the computing overhead)
            div_val=1,  # Disables adaptative input (embeddings), because the embeddings are managed by TabularFeatures
            **kwargs,
        )
