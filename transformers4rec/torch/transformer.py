import transformers


class T4RecConfig:
    _model = None

    def to_model(self):
        return self._model(self)


class XLNetConfig(T4RecConfig, transformers.XLNetConfig):
    _model = transformers.XLNetModel

    @classmethod
    def for_rec(cls, d_model, n_head, n_layer,
                hidden_act='gelu', initializer_range=0.01,
                layer_norm_eps=0.03, dropout=0.3, pad_token=0,
                log_attention_weights=True, **kwargs):
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
