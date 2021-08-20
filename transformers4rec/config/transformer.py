import transformers

from ..utils.registry import Registry

transformer_registry: Registry = Registry("transformers")


class T4RecConfig:
    def to_huggingface_torch_model(self):
        model_cls = transformers.MODEL_MAPPING[self.transformers_config_cls]

        return model_cls(self)

    def to_torch_model(self, input_features, *prediction_task, **kwargs):
        from .. import torch as torch4rec

        if not isinstance(input_features, torch4rec.SequentialTabularFeatures):
            raise ValueError("`input_features` must an instance of SequentialTabularFeatures")
        if not all(isinstance(t, torch4rec.PredictionTask) for t in prediction_task):
            raise ValueError(
                "`task` is of the wrong type, please provide one or multiple "
                "instance(s) of PredictionTask"
            )

        body = torch4rec.SequentialBlock(
            input_features, torch4rec.TransformerBlock(self, masking=input_features.masking)
        )

        return torch4rec.Head(body, *prediction_task, **kwargs).to_model(**kwargs)

    def to_huggingface_tf_model(self):
        model_cls = transformers.TF_MODEL_MAPPING[self.transformers_config_cls]

        return model_cls(self)

    @property
    def transformers_config_cls(self):
        return self.__class__.__bases__[1]

    @classmethod
    def build(cls, *args, **kwargs):
        raise NotImplementedError


@transformer_registry.register("reformer")
class ReformerConfig(T4RecConfig, transformers.ReformerConfig):
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
        log_attention_weights=True,
        axial_pos_shape_first_dim=4,
        **kwargs
    ):
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
            **kwargs
        )


@transformer_registry.register("gtp2")
class GPT2Config(T4RecConfig, transformers.GPT2Config):
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
        log_attention_weights=True,
        **kwargs
    ):
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
            **kwargs
        )


@transformer_registry.register("longformer")
class LongformerConfig(T4RecConfig, transformers.LongformerConfig):
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
        log_attention_weights=True,
        **kwargs
    ):
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
            **kwargs
        )


@transformer_registry.register("electra")
class ElectraConfig(T4RecConfig, transformers.ElectraConfig):
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
        log_attention_weights=True,
        **kwargs
    ):
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
            **kwargs
        )


@transformer_registry.register("albert")
class AlbertConfig(T4RecConfig, transformers.AlbertConfig):
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
        log_attention_weights=True,
        **kwargs
    ):
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
            **kwargs
        )


@transformer_registry.register("xlnet")
class XLNetConfig(T4RecConfig, transformers.XLNetConfig):
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
        log_attention_weights=True,
        **kwargs
    ):
        return cls(
            d_model=d_model,
            d_inner=d_model * 4,
            n_layer=n_layer,
            n_head=n_head,
            ff_activation=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs
        )
