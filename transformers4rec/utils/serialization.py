import io
# pickle is not secure, but but this whole file is a wrapper to make it
# possible to mitigate the primary risk of code injection via pickle.
import pickle  # nosec B403
# import cloudpickle

# These are the base classes that are generally serialized by the ZeroMQ IPC.
# If a class is needed by ZMQ routinely it should be added here. If
# it is only needed in a single instance the class can be added at runtime
# using register_approved_ipc_class.
BASE_SERIALIZATION_CLASSES = {
    "builtins": [
        "Exception", "ValueError", "NotImplementedError", "AttributeError",
        "AssertionError"
    ],  # each Exception Error class needs to be added explicitly
    "collections": ["OrderedDict", "defaultdict"],
    "datetime": ["timedelta"],
    "pathlib": ["PosixPath"],
    "functools": ["partial"],
    "transformers4rec.torch.model.base": ["Model", "Head", "PredictionTask"],
    "transformers4rec.torch.block.base": ["SequentialBlock"],
    "transformers4rec.torch.features.sequence": ["TabularSequenceFeatures", "SequenceEmbeddingFeatures"],
    "transformers4rec.torch.features.continuous": ["ContinuousFeatures"],
    "transformers4rec.torch.features.embedding": ["FeatureConfig", "EmbeddingConfig", "TableConfig", "PretrainedEmbeddingFeatures"],
    "transformers4rec.torch.features.sparse": ["SparseFeatures"],
    "transformers4rec.torch.features.tabular": ["TabularFeatures"],
    "transformers4rec.torch.tabular.base": ["FilterFeatures", "AsTabular"],
    "transformers4rec.torch.tabular.aggregation": ["ConcatFeatures"],
    "transformers4rec.torch.block.base": ["Block", "SequentialBlock"],
    "transformers4rec.torch.block.mlp": ["DenseBlock"],
    "transformers4rec.torch.block.transformer": ["TransformerBlock"],
    "transformers4rec.torch.masking": ["CausalLanguageModeling"],
    "transformers4rec.torch.model.base": ["forward_to_prediction_fn", "Model", "Head"],
    "transformers4rec.torch.model.prediction_task": ["NextItemPredictionTask", "_NextItemPredictionTask"],
    "transformers4rec.torch.ranking_metric": ["NDCGAt", "DCGAt", "AvgPrecisionAt", "PrecisionAt", "RecallAt"],
    "transformers4rec.config.transformer": ["XLNetConfig"],
    "torch.nn.modules.container": ["ModuleList","ModuleDict"],
    "torch.nn.modules.loss": ["CrossEntropyLoss"],
    "merlin_standard_lib.schema.schema": ["Schema", "ColumnSchema"],
    "merlin_standard_lib.proto.schema_bp": [
        "FeaturePresence", "FeaturePresenceWithinGroup", "FeatureType", "FixedShape", "ValueCount", "ValueCountList", 
        "IntDomain", "FloatDomain", "StringDomain", "BoolDomain", "StructDomain", "NaturalLanguageDomain", 
        "FeatureCoverageConstraints", "SequenceLengthConstraints", "ImageDomain", "MIDDomain", "URLDomain", 
        "TimeDomain", "TimeOfDayDomain", "DistributionConstraints", "Annotation", "FeatureComparator", "InfinityNorm", 
        "JensenShannonDivergence", "UniqueConstraints", "DatasetConstraints", "NumericValueComparator"],
    "torch.nn.init": ["kaiming_normal_", "kaiming_uniform_", "xavier_normal_", "xavier_uniform_", "uniform_", "normal_", "zeros_", "ones_"],
    "torch._utils": ["_rebuild_tensor_v2", "_rebuild_parameter"],
    "torch": ["Size", "device"],
    "torch.storage": ["_load_from_bytes"],
    "torch._C._nn": ["gelu"],
    "torch.nn.module": ["Module"],
    "torch.nn.modules.activation": ["ReLU", "Sigmoid", "Tanh"],
    "torch.nn.modules.linear": ["Linear", "Identity"],
    "torch.nn.modules.conv": ["Conv1d", "Conv2d", "Conv3d"],
    "torch.nn.modules.pooling": ["MaxPool1d", "MaxPool2d", "MaxPool3d"],
    "torch.nn.modules.normalization": ["BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm"],
    "torch.nn.modules.dropout": ["Dropout", "Dropout2d", "Dropout3d"],
    "torch.nn.modules.rnn": ["RNN", "LSTM", "GRU"],
    "torch.nn.modules.sparse": ["Embedding"],
    "torch.optim.adam": ["Adam"],
    "torchmetrics.metric": ["jit_distributed_available"],
    "torchmetrics.utilities.data": ["dim_zero_cat"],
    "transformers.models.xlnet.modeling_xlnet": ["XLNetModel", "XLNetLayer", "XLNetRelativeAttention", "XLNetFeedForward"],
    "transformers.activations": ["GELUActivation"],
    "transformers.modeling_utils": ["SequenceSummary"],
    "builtins": ["getattr"],

}


def _register_class(dict, obj):
    name = getattr(obj, '__qualname__', None)
    if name is None:
        name = obj.__name__
    module = pickle.whichmodule(obj, name)
    if module not in BASE_SERIALIZATION_CLASSES.keys():
        BASE_SERIALIZATION_CLASSES[module] = []
    BASE_SERIALIZATION_CLASSES[module].append(name)


def register_approved_ipc_class(obj):
    _register_class(BASE_SERIALIZATION_CLASSES, obj)


class Unpickler(pickle.Unpickler):

    def __init__(self, *args, approved_imports={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.approved_imports = approved_imports
        self.approved_imports.update(BASE_SERIALIZATION_CLASSES)

    # only import approved classes, this is the security boundary.
    def find_class(self, module, name):
        if name not in self.approved_imports.get(module, []):
            # If this is triggered when it shouldn't be, then the module
            # and class should be added to the approved_imports. If the class
            # is being used as part of a routine scenario, then it should be added
            # to the appropriate base classes above.
            raise ValueError(f"Import {module} | {name} is not allowed")
        return super().find_class(module, name)


# these are taken from the pickle module to allow for this to be a drop in replacement
# source: https://github.com/python/cpython/blob/3.13/Lib/pickle.py
# dump and dumps are just aliases because the serucity controls are on the deserialization
# side. However they are included here so that in the future if a more secure serialization
# soliton is identified, it can be added with less impact to the rest of the application.
dump = pickle.dump  # nosec B301
dumps = pickle.dumps  # nosec B301


def load(file,
         *,
         fix_imports=True,
         encoding="ASCII",
         errors="strict",
         buffers=None,
         approved_imports={}):
    return Unpickler(file,
                     fix_imports=fix_imports,
                     buffers=buffers,
                     encoding=encoding,
                     errors=errors,
                     approved_imports=approved_imports).load()


def loads(s,
          /,
          *,
          fix_imports=True,
          encoding="ASCII",
          errors="strict",
          buffers=None,
          approved_imports={}):
    if isinstance(s, str):
        raise TypeError("Can't load pickle from unicode string")
    file = io.BytesIO(s)
    return Unpickler(file,
                     fix_imports=fix_imports,
                     buffers=buffers,
                     encoding=encoding,
                     errors=errors,
                     approved_imports=approved_imports).load()