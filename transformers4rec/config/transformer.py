

class T4RecConfig:
    def to_torch_model(self):
        from transformers import MODEL_MAPPING

        return MODEL_MAPPING[self]

    def to_tf_model(self):
        from transformers import TF_MODEL_MAPPING

        return TF_MODEL_MAPPING[self]

    @classmethod
    def for_rec(cls, *args, **kwargs):
        raise NotImplementedError
