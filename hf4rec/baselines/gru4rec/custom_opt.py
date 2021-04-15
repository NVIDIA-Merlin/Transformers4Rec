import theano
from theano import tensor, config
from theano.gpuarray.subtensor import GpuAdvancedSubtensor1
from theano.gpuarray.opt import register_opt, op_lifter, register_opt2
from custom_theano_ops import GpuAdvancedSubtensor1_fast

def remove_optimization(optimizer, name, *tags):
    obj = optimizer.__db__[name].copy().pop()
    optimizer.remove_tags(name, *tags)
    optimizer.__db__[obj.__class__.__name__].remove(obj)
    optimizer._names.remove(name)
    del(optimizer.__db__[name])

def get_tags(optimizer, name):
    obj = optimizer.__db__[name].copy().pop()
    tags = []
    for k, v in optimizer.__db__.items():
        if (obj in v) and (k != name) and (k != obj.__class__.__name__):
            tags.append(k)
    return sorted(tags)

tags = get_tags(theano.gpuarray.opt.gpu_optimizer, 'local_gpua_advanced_subtensor1')
remove_optimization(theano.gpuarray.opt.gpu_optimizer, 'local_gpua_advanced_subtensor1', *tags)

tags = get_tags(theano.gpuarray.opt.gpu_optimizer2, 'local_gpua_advanced_subtensor1')
remove_optimization(theano.gpuarray.opt.gpu_optimizer2, 'local_gpua_advanced_subtensor1', *tags)

@register_opt('fast_compile')
@op_lifter([tensor.AdvancedSubtensor1])
@register_opt2([tensor.AdvancedSubtensor1], 'fast_compile')
def local_gpua_advanced_subtensor1(op, context_name, inputs, outputs):
    x, ilist = inputs
    if (x.ndim != 2 or config.deterministic == 'more'):
        return GpuAdvancedSubtensor1()
    else:
        return GpuAdvancedSubtensor1_fast()
