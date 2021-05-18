from theano import tensor, gof, Op, config
from theano.gof import ParamsType
from theano.gradient import grad_not_implemented
import theano.tensor as T
from theano.gpuarray.subtensor import GpuAdvancedSubtensor1
from theano.scalar import bool as bool_t, int32 as int_t, uint32 as size_t

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

from theano.gpuarray.type import GpuArrayType, gpu_context_type, get_context
from theano.gpuarray.basic_ops import (as_gpuarray_variable, HideC, GpuKernelBase, Kernel, gpuarray_helper_inc_dir, infer_context_name, gpu_contiguous)
from theano.gpuarray.fp16_help import write_w, load_w, work_dtype

class GpuExtractDiag2D(GpuKernelBase, Op):
    """
    Extracting diagonal of a 2D matrix on the GPU.

    """
    __props__ = ('context_name', 'keepdims')
    _f16_ok = True
    params_type = ParamsType(context=gpu_context_type, keepdims=bool_t)

    def __init__(self, context_name=None, keepdims=False):
        self.context_name = context_name
        self.keepdims = keepdims

    def get_params(self, node):
        return self.params_type.get_params(self, context=get_context(self.context_name), keepdims=self.keepdims)

    def make_node(self, x, k=0): #TODO: dtype check
        x = as_gpuarray_variable(x, context_name=self.context_name)
        k = tensor.as_tensor_variable(k)
        assert x.ndim == 2
        assert k.ndim == 0
        broadcastable = (False,True) if self.keepdims else (False,)
        otype = GpuArrayType(dtype=x.type.dtype, broadcastable=broadcastable, context_name=self.context_name)
        return gof.Apply(self, [x, k], [otype()])

    def infer_shape(self, node, in_shapes):
        in_shape, _ = in_shapes
        dim1 = in_shape[0]
        dim2 = in_shape[1]
        k = node.inputs[1]
        diag_size = T.switch(T.ge(k, 0), T.clip(dim2 - k, 0, dim1), T.clip(dim1 + k, 0, dim2))
        if self.keepdims:
            diag_size = (diag_size, 1)
        else:
            diag_size = (diag_size,)
        return [diag_size]

    def grad(self, inp, grads):
        return [GpuAllocDiag2D()(grads[0], inp[1], *(inp[0].shape)), grad_not_implemented(self, 1, inp[1])]

    def gpu_kernels(self, node, name):
        dtype_x = node.inputs[0].dtype
        type_x = gpuarray.dtype_to_ctype(dtype_x)
        dtype_y = node.outputs[0].dtype
        type_y = gpuarray.dtype_to_ctype(dtype_y)
        work_x = gpuarray.dtype_to_ctype(work_dtype(dtype_x))
        load_x = load_w(dtype_x)
        write_y = write_w(dtype_y)
        code = """
        #include "cluda.h"
        KERNEL void extract(const ga_ssize stridesX0, const ga_ssize stridesX1, GLOBAL_MEM %(type_x)s *x, ga_size x_off, const ga_ssize stridesY0, GLOBAL_MEM %(type_y)s *y, ga_size y_off, ga_ssize k, ga_size l) {
            x = (GLOBAL_MEM %(type_x)s *)(((GLOBAL_MEM char *)x) + x_off);
            y = (GLOBAL_MEM %(type_y)s *)(((GLOBAL_MEM char *)y) + y_off);
            ga_ssize coff = max(k, (ga_ssize) 0);
            ga_ssize roff = -min(k, (ga_ssize) 0);
            ga_size index = GID_0 * LDIM_0 + LID_0;
            if (index < l) {
                %(work_x)s t = %(load_x)s(x[(index + roff) * stridesX0 + (index + coff) * stridesX1]);
                y[index * stridesY0] = %(write_y)s(t);
            }
        }""" % dict(type_x=type_x, type_y=type_y, work_x=work_x, load_x=load_x, write_y=write_y, name=name)
        return [Kernel(
                code=code, name="extract",
                params=[gpuarray.SSIZE, gpuarray.SSIZE, gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SIZE],
                flags=Kernel.get_flags(dtype_x, dtype_y),
                objvar='k_extract_' + name)]

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray_helper.h>', '<gpuarray/types.h>']

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def c_code(self, node, name, inp, out, sub): #TODO: fix error msg
        x, k = inp
        y, = out
        fail = sub['fail']
        params = sub['params']
        typecode = pygpu.gpuarray.dtype_to_typecode(node.inputs[0].dtype)
        kname = self.gpu_kernels(node, name)[0].objvar
        s = """
        int err;
        size_t* dims = (size_t*)PyGpuArray_DIMS((PyGpuArrayObject*)%(x)s);
        size_t k = ((dtype_%(k)s*)PyArray_DATA(%(k)s))[0];
        size_t col_off = (size_t) (k > 0?k:0);
        size_t row_off = (size_t) (k < 0?-k:0);
        size_t diag_size = (size_t) std::max((ssize_t) std::min((ssize_t)dims[0] - (ssize_t)row_off, (ssize_t)dims[1] - (ssize_t)col_off), (ssize_t) 0);
        size_t ls = std::min(diag_size, (size_t) 1024);
        size_t gs = (diag_size + ls - 1) / ls;
        size_t ndims = %(params)s->keepdims ? 2 : 1;
        size_t out_dims[ndims];
        out_dims[0] = diag_size;
        if (ndims == 2) {
            out_dims[1] = 1;
        }

        size_t itemsize_x = 1;
        size_t itemsize_y = 1;
        ssize_t stridesX0 = 1;
        ssize_t stridesX1 = 1;
        ssize_t stridesY0 = 1;

        if (%(y)s == NULL || %(y)s->ga.nd != ndims || %(y)s->ga.dimensions[0] != diag_size || (ndims > 1 && %(y)s->ga.dimensions[1] != 1)) {
            Py_CLEAR(%(y)s);
            %(y)s = pygpu_empty(ndims, out_dims, %(typecode)s, GA_C_ORDER, %(params)s->context, Py_None);
        }
        if (%(y)s == NULL) {
            %(fail)s
        }

        itemsize_x = GpuArray_ITEMSIZE(&%(x)s->ga);
        itemsize_y = GpuArray_ITEMSIZE(&%(y)s->ga);
        stridesX0 = PyGpuArray_STRIDES(%(x)s)[0] / itemsize_x;
        stridesX1 = PyGpuArray_STRIDES(%(x)s)[1] / itemsize_x;
        stridesY0 = PyGpuArray_STRIDES(%(y)s)[0] / itemsize_y;

        if (row_off < dims[0] && col_off < dims[1]) {
            err = extract_call(1, &gs, &ls, 0, stridesX0, stridesX1, %(x)s->ga.data, %(x)s->ga.offset, stridesY0, %(y)s->ga.data, %(y)s->ga.offset, k, diag_size);
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError, "gpuarray error: kExtract: %%s. n%%lu, m=%%lu.", GpuKernel_error(&%(kname)s, err), (unsigned long)dims[0], (unsigned long)dims[1]);
                %(fail)s;
            }
        } else {
            %(fail)s;
        }
        """ % locals()
        return s

    def c_code_cache_version(self):
        return (1,)

class GpuAllocDiag2D(GpuKernelBase, Op):
    """
    Making a diagonal matrix from a vector on GPU

    """
    __props__ = ('context_name',)
    _f16_ok = True

    def __init__(self, context_name=None):
        self.context_name = context_name

    def get_params(self, node):
        return get_context(self.context_name)

    def make_node(self, x, k=0, n=0, m=0): #TODO: dtype check
        x = as_gpuarray_variable(x, context_name=self.context_name)
        k = tensor.as_tensor_variable(k)
        n = tensor.as_tensor_variable(n)
        m = tensor.as_tensor_variable(m)
        assert x.ndim == 2 or x.ndim == 1
        assert k.ndim == 0
        assert n.ndim == 0
        assert m.ndim == 0
        otype = GpuArrayType(dtype=x.type.dtype, broadcastable=(False,False), context_name=self.context_name)
        return gof.Apply(self, [x, k, n, m], [otype()])

    def infer_shape(self, node, in_shapes):
        in_shape, _, _, _ = in_shapes
        k, n, m = node.inputs[1:]
        dim_in = in_shape[0]
        dim_out1 = T.maximum(T.switch(T.ge(k,0), dim_in, dim_in-k), n)
        dim_out2 = T.maximum(T.switch(T.ge(k,0), dim_in+k, dim_in), m)
        return [(dim_out1, dim_out2)]

    def grad(self, inp, grads):
        return [GpuExtractDiag2D(keepdims=(inp[0].ndim==2))(grads[0], inp[1])] + [grad_not_implemented(self, i, inp[i]) for i in range(1,4)]

    def gpu_kernels(self, node, name):
        dtype_x = node.inputs[0].dtype
        type_x = gpuarray.dtype_to_ctype(dtype_x)
        dtype_y = node.outputs[0].dtype
        type_y = gpuarray.dtype_to_ctype(dtype_y)
        work_x = gpuarray.dtype_to_ctype(work_dtype(dtype_x))
        load_x = load_w(dtype_x)
        write_y = write_w(dtype_y)
        code = """
        #include "cluda.h"
        KERNEL void dalloc(const ga_ssize stridesX0, GLOBAL_MEM %(type_x)s *x, ga_size x_off, const ga_ssize stridesY0, const ga_ssize stridesY1, GLOBAL_MEM %(type_y)s *y, ga_size y_off, ga_ssize k, ga_size l) {
            x = (GLOBAL_MEM %(type_x)s *)(((GLOBAL_MEM char *)x) + x_off);
            y = (GLOBAL_MEM %(type_y)s *)(((GLOBAL_MEM char *)y) + y_off);
            ga_ssize coff = max(k, (ga_ssize) 0);
            ga_ssize roff = -min(k, (ga_ssize) 0);
            ga_size index = GID_0 * LDIM_0 + LID_0;
            if (index < l) {
                %(work_x)s t = %(load_x)s(x[index * stridesX0]);
                y[(index + roff) * stridesY0 + (index + coff) * stridesY1] = %(write_y)s(t);
            }
        }""" % dict(type_x=type_x, type_y=type_y, work_x=work_x, load_x=load_x, write_y=write_y, name=name)
        return [Kernel(
                code=code, name="dalloc",
                params=[gpuarray.SSIZE, gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE, gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE, gpuarray.SIZE],
                flags=Kernel.get_flags(dtype_x, dtype_y),
                objvar='k_dalloc_' + name)]

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray_helper.h>', '<gpuarray/types.h>']

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def c_code(self, node, name, inp, out, sub):  #TODO: fix error msgs
        x, k, n, m = inp
        y, = out
        fail = sub['fail']
        ctx = sub['params']
        typecode = pygpu.gpuarray.dtype_to_typecode(node.inputs[0].dtype)
        kname = self.gpu_kernels(node, name)[0].objvar
        s = """
        int err;
        size_t ndims = (size_t)PyGpuArray_NDIM((PyGpuArrayObject*)%(x)s);
        size_t* in_dims = (size_t*)PyGpuArray_DIMS((PyGpuArrayObject*)%(x)s);
        size_t l = in_dims[0];
        size_t ls = std::min(l, (size_t)1024);
        size_t gs = (l + ls - 1) / ls;
        size_t k = ((dtype_%(k)s*)PyArray_DATA(%(k)s))[0];
        size_t n = ((dtype_%(n)s*)PyArray_DATA(%(n)s))[0];
        size_t m = ((dtype_%(m)s*)PyArray_DATA(%(m)s))[0];
        size_t out_dims[2] = {std::max(k < 0 ? (size_t)l-k : l, n), std::max(k > 0 ? (size_t)l+k : l, m)};

        size_t itemsize_x = 1;
        size_t itemsize_y = 1;
        ssize_t stridesX0 = 1;
        ssize_t stridesY0 = 1;
        ssize_t stridesY1 = 1;
        
        if ((ndims == 2) && (in_dims[1] != 1)) {
            PyErr_Format(PyExc_RuntimeError, "If the input has 2 dimensions the second dimension must be of size 1. Input shape: (%%lu, %%lu)", (unsigned long)in_dims[0], (unsigned long)in_dims[1]);
            %(fail)s
        }

        Py_CLEAR(%(y)s);
        %(y)s = pygpu_zeros(2, out_dims, %(typecode)s, GA_C_ORDER, %(ctx)s, Py_None); //theano can reuse this space, thus we have to make sure to fill it with zeros every time

        if (%(y)s == NULL) {
            PyErr_Format(PyExc_RuntimeError, "Failed to allocate array for the output.");
            %(fail)s
        }

        itemsize_x = GpuArray_ITEMSIZE(&%(x)s->ga);
        itemsize_y = GpuArray_ITEMSIZE(&%(y)s->ga);
        stridesX0 = PyGpuArray_STRIDES(%(x)s)[0] / itemsize_x;
        stridesY0 = PyGpuArray_STRIDES(%(y)s)[0] / itemsize_y;
        stridesY1 = PyGpuArray_STRIDES(%(y)s)[1] / itemsize_y;

        err = dalloc_call(1, &gs, &ls, 0, stridesX0, %(x)s->ga.data, %(x)s->ga.offset, stridesY0, stridesY1, %(y)s->ga.data, %(y)s->ga.offset, k, l);
        if (err != GA_NO_ERROR) {
            PyErr_Format(PyExc_RuntimeError, "gpuarray error: kAlloc: %%s. n%%lu, m=%%lu.", GpuKernel_error(&%(kname)s, err), (unsigned long)out_dims[0], (unsigned long)out_dims[1]);
            %(fail)s;
        }

        """ % locals()
        return s

    def c_code_cache_version(self):
        return (1,)

class GpuBinarySearchSorted(GpuKernelBase, Op):
    """
    Searchsorted on GPU

    """
    __props__ = ('context_name', 'dtype_int64')
    _f16_ok = True
    params_type = ParamsType(context=gpu_context_type, dtype_int64=bool_t)

    def __init__(self, context_name=None, dtype_int64=False):
        self.context_name = context_name
        self.dtype_int64 = dtype_int64

    def get_params(self, node):
        return self.params_type.get_params(self, context=get_context(self.context_name), dtype_int64=self.dtype_int64)

    def make_node(self, d, x):
        d = as_gpuarray_variable(d, context_name=self.context_name)
        x = as_gpuarray_variable(x, context_name=self.context_name)
        assert d.ndim == 1
        assert x.ndim == 1
        broadcastable = (False,)
        otype = GpuArrayType(dtype='int64' if self.dtype_int64 else 'int32', broadcastable=broadcastable, context_name=self.context_name)
        return gof.Apply(self, [d, x], [otype()])

    def infer_shape(self, node, in_shapes):
        _, x_shape = in_shapes
        return [x_shape]

    def grad(self, inp, grads):
        return [grad_not_implemented(self, i, inp[i]) for i in range(2)]

    def gpu_kernels(self, node, name):
        dtype_d = node.inputs[0].dtype
        type_d = gpuarray.dtype_to_ctype(dtype_d)
        dtype_x = node.inputs[1].dtype
        type_x = gpuarray.dtype_to_ctype(dtype_x)
        dtype_y = node.outputs[0].dtype
        type_y = gpuarray.dtype_to_ctype(dtype_y)
        work_d = gpuarray.dtype_to_ctype(work_dtype(dtype_d))
        load_d = load_w(dtype_d)
        work_x = gpuarray.dtype_to_ctype(work_dtype(dtype_x))
        load_x = load_w(dtype_x)
        code = """
        #include "cluda.h"
        KERNEL void binsearchsorted(const ga_ssize stridesD0, GLOBAL_MEM %(type_d)s *d, ga_size d_off, const ga_ssize stridesX0, GLOBAL_MEM %(type_x)s *x, ga_size x_off, const ga_ssize stridesY0, GLOBAL_MEM %(type_y)s *y, ga_size y_off, ga_size lx, ga_ssize ld) {
            d = (GLOBAL_MEM %(type_d)s *)(((GLOBAL_MEM char *)d) + d_off);
            x = (GLOBAL_MEM %(type_x)s *)(((GLOBAL_MEM char *)x) + x_off);
            y = (GLOBAL_MEM %(type_y)s *)(((GLOBAL_MEM char *)y) + y_off);
            ga_size index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < lx) {
                ga_long a = 0;
                ga_long b = (ga_long)(ld - 1);
                %(work_d)s minval = %(load_d)s(d[a]);
                %(work_d)s maxval = %(load_d)s(d[b * stridesD0]);
                %(work_x)s val = %(load_x)s(x[index * stridesX0]);
                if (val > maxval) {
                    a = (ga_long)ld;
                    b = (ga_long)ld;
                } else if (val <= minval) {
                    a = 0;
                    b = 0;
                }
                while (b - a > 0) {
                    ga_long h = (b + a) / 2;
                    %(work_d)s t = %(load_d)s(d[h * stridesD0]);
                    if (val < t) {
                        b = h;
                    } else {
                        a = h + 1;
                    }
                }
                y[index * stridesY0] = b;
            }
        }""" % dict(type_d=type_d, type_x=type_x, type_y=type_y, work_d=work_d, load_d=load_d, work_x=work_x, load_x=load_x, name=name)
        return [Kernel(
                code=code, name="binsearchsorted",
                params=[gpuarray.SSIZE, gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE, gpuarray.SSIZE],
                flags=Kernel.get_flags(dtype_d, dtype_x, dtype_y),
                objvar='k_binsearchsorted_' + name)]

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray_helper.h>', '<gpuarray/types.h>']

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def c_code(self, node, name, inp, out, sub): #TODO: fix error msg
        d, x = inp
        y, = out
        fail = sub['fail']
        params = sub['params']
        typecode = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        kname = self.gpu_kernels(node, name)[0].objvar
        s = """
        int err;
        size_t dimd = ((size_t*)PyGpuArray_DIMS((PyGpuArrayObject*)%(d)s))[0];
        size_t dimx = ((size_t*)PyGpuArray_DIMS((PyGpuArrayObject*)%(x)s))[0];
        size_t ls = 1024;
        size_t gs = (dimx / 1024) + 1;
        size_t out_dims[1] = {dimx};

        size_t itemsize_d = 1;
        size_t itemsize_x = 1;
        size_t itemsize_y = 1;
        ssize_t stridesD0 = 1;
        ssize_t stridesX0 = 1;
        ssize_t stridesY0 = 1;

        if (%(y)s == NULL || %(y)s->ga.nd != 1 || %(y)s->ga.dimensions[0] != dimx) {
            Py_CLEAR(%(y)s);
            %(y)s = pygpu_zeros(1, out_dims, %(typecode)s, GA_C_ORDER, %(params)s->context, Py_None);
        }
        if (%(y)s == NULL) {
            %(fail)s
        }

        itemsize_d = GpuArray_ITEMSIZE(&%(d)s->ga);
        itemsize_x = GpuArray_ITEMSIZE(&%(x)s->ga);
        itemsize_y = GpuArray_ITEMSIZE(&%(y)s->ga);
        stridesD0 = PyGpuArray_STRIDES(%(d)s)[0] / itemsize_d;
        stridesX0 = PyGpuArray_STRIDES(%(x)s)[0] / itemsize_x;
        stridesY0 = PyGpuArray_STRIDES(%(y)s)[0] / itemsize_y;
        err = binsearchsorted_call(1, &gs, &ls, 0, stridesD0, %(d)s->ga.data, %(d)s->ga.offset, stridesX0, %(x)s->ga.data, %(x)s->ga.offset, stridesY0, %(y)s->ga.data, %(y)s->ga.offset, dimx, (ssize_t)dimd);
        if (err != GA_NO_ERROR) {
            PyErr_Format(PyExc_RuntimeError, "gpuarray error: kExtract: %%s. n%%lu, m=%%lu.", GpuKernel_error(&%(kname)s, err), (unsigned long)dimx, (unsigned long)dimd);
            %(fail)s;
        }
        """ % locals()
        return s

    def c_code_cache_version(self):
        return (1,)

class GpuAdvancedSubtensor1_fast(GpuKernelBase, GpuAdvancedSubtensor1):
    """
    Implement a faster version AdvancedSubtensor1 on the gpu for 2D tensors

    """
    _f16_ok = True

    def make_node(self, x, ilist):
        ctx_name = infer_context_name(x, ilist)
        x_ = as_gpuarray_variable(x, ctx_name)
        ilist_ = as_gpuarray_variable(ilist, ctx_name)

        if ilist_.type.dtype not in tensor.integer_dtypes:
            raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')
        return gof.Apply(self, [x_, ilist_], [x_.type()])

    def perform(self, node, inp, out, params):
        return super(GpuAdvancedSubtensor1_fast, self).perform(node, inp, out)

    def c_code_cache_version(self):
        return (1,)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray_helper.h>',
                '<gpuarray/types.h>']

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def c_code(self, node, name, inputs, outputs, sub):
        if (node.inputs[0].ndim != 2):
            raise NotImplementedError("This case does not have C code yet.")

        return """
int err;
if (%(out)s == NULL || !GpuArray_IS_C_CONTIGUOUS(&%(out)s->ga) ||
    %(out)s->ga.dimensions[0] != %(idx)s->ga.dimensions[0] ||
    %(out)s->ga.nd != %(v)s->ga.nd || %(out)s->ga.dimensions[1] != %(v)s->ga.dimensions[1]) {
  size_t tmp;
  Py_XDECREF(%(out)s);

  /* This is a dirty hack to avoid an extra alloc */
  tmp = %(v)s->ga.dimensions[0];
  %(v)s->ga.dimensions[0] = %(idx)s->ga.dimensions[0];
  %(out)s = pygpu_empty(%(v)s->ga.nd, %(v)s->ga.dimensions, %(v)s->ga.typecode,
                        GA_C_ORDER, %(v)s->context, Py_None);
  if (%(out)s == NULL) {
    %(fail)s;
  }
  %(v)s->ga.dimensions[0] = tmp; // Don't remove this line
}
if (GpuArray_vector_select_fast(%(out)s, %(v)s, %(idx)s)) {
  %(fail)s
}
        """ % dict(v=inputs[0], idx=inputs[1], out=outputs[0], fail=sub['fail'])

    def gpu_kernels(self, node, nodename):
        CHARMAP = dict(int32='i', uint32='I',
                       int64='l', uint64='L',
                       float16='e', float32='f', float64='d')
        dtype_in = node.inputs[0].dtype
        dtype_out = node.outputs[0].dtype
        dtype_idx = node.inputs[1].dtype
        type_in = gpuarray.dtype_to_ctype(dtype_in)
        type_out = gpuarray.dtype_to_ctype(dtype_out)
        type_idx = gpuarray.dtype_to_ctype(dtype_idx)
        flags = Kernel.get_flags(dtype_in, dtype_out, dtype_idx)
        kname = "k_vector_select_fast"
        k_var = "k_vector_select_fast_" + nodename
        code = """#include "cluda.h"
        KERNEL void k_vector_select_fast(const ga_size numRowsOut,
                                      const ga_size numColsOut,
                                      const ga_ssize stridesOut0,
                                      const ga_ssize stridesOut1,
                                      GLOBAL_MEM %(type_out)s *Out,
                                      const ga_size offset_Out,
                                      const ga_size numRowsIn,
                                      const ga_size numColsIn,
                                      const ga_ssize stridesIn0,
                                      const ga_ssize stridesIn1,
                                      GLOBAL_MEM %(type_in)s *In,
                                      const ga_size offset_In,
                                      const ga_size numIndices,
                                      const ga_ssize stridesIndices,
                                      GLOBAL_MEM %(type_idx)s *indices_arr,
                                      const ga_size offset_indices_arr,
                                      GLOBAL_MEM ga_int *err)
        {
             Out = (GLOBAL_MEM %(type_out)s *)(((GLOBAL_MEM char *)Out)+offset_Out);
             In = (GLOBAL_MEM %(type_in)s *)(((GLOBAL_MEM char *)In)+offset_In);
             indices_arr = (GLOBAL_MEM %(type_idx)s *)(((GLOBAL_MEM char *)indices_arr)+offset_indices_arr);

             for (ga_int i = GID_0; i < numIndices; i += GDIM_0)
             {
                  for (ga_int j = LID_0; j < numColsIn; j += LDIM_0)
                  {
                      ga_ssize in_row = indices_arr[i * stridesIndices];
                      if (in_row < 0)
                          in_row += numRowsIn;
                      ga_ssize out_row = i;
                      if (in_row < numRowsIn && in_row >= 0) {
                        Out[(out_row * stridesOut0) + (j * stridesOut1)] = In[(in_row * stridesIn0) + (j * stridesIn1)];
                      } else {
                        *err = 1;
                      }
                  }
             }
             return;
        }
        """ % dict(type_in=type_in, type_out=type_out, type_idx=type_idx,
                   tc=CHARMAP[dtype_in])
        from pygpu.gpuarray import SIZE, SSIZE
        params = [
            SIZE, SIZE, SSIZE, SSIZE, gpuarray.GpuArray, SIZE,
            SIZE, SIZE, SSIZE, SSIZE, gpuarray.GpuArray, SIZE,
            SIZE, SSIZE, gpuarray.GpuArray, SIZE,
            gpuarray.GpuArray]
        return [Kernel(code=code, name=kname, params=params,
                       flags=flags, objvar=k_var)]

    def c_support_code_struct(self, node, nodename):
        return super(GpuAdvancedSubtensor1_fast, self).c_support_code_struct(node, nodename) + """
        int GpuArray_vector_select_fast(PyGpuArrayObject* py_out,
                                     PyGpuArrayObject* py_in,
                                     PyGpuArrayObject* indices_arr)
        {
            size_t threads_per_block = std::min(PyGpuArray_DIMS(py_out)[1], (size_t)256);
            size_t n_blocks = std::min(PyGpuArray_SIZE(indices_arr), (size_t)4096);
            gpudata *errbuf;
            int err, kerr = 0;
            size_t itemsize_out = GpuArray_ITEMSIZE(&py_out->ga);
            size_t itemsize_in = GpuArray_ITEMSIZE(&py_in->ga);
            size_t itemsize_idx = GpuArray_ITEMSIZE(&indices_arr->ga);

            if (threads_per_block > 0 && n_blocks > 0) {
              err = gpudata_property(py_out->ga.data,
                                     GA_CTX_PROP_ERRBUF, &errbuf);
              if (err != GA_NO_ERROR) {
                PyErr_SetString(PyExc_RuntimeError, "Can't fetch error buffer");
                return 1;
              }

              err = k_vector_select_fast_call(
        1, &n_blocks, &threads_per_block, 0,
        PyGpuArray_DIMS(py_out)[0],
        PyGpuArray_DIMS(py_out)[1],
        PyGpuArray_STRIDES(py_out)[0] / itemsize_out,
        PyGpuArray_STRIDES(py_out)[1] / itemsize_out,
        py_out->ga.data,
        py_out->ga.offset,
        PyGpuArray_DIMS(py_in)[0],
        PyGpuArray_DIMS(py_in)[1],
        PyGpuArray_DIMS(py_in)[0] == 1 ? 0 : PyGpuArray_STRIDES(py_in)[0] / itemsize_in,
        PyGpuArray_DIMS(py_in)[1] == 1 ? 0 : PyGpuArray_STRIDES(py_in)[1] / itemsize_in,
        py_in->ga.data,
        py_in->ga.offset,
        PyGpuArray_DIMS(indices_arr)[0],
        PyGpuArray_STRIDES(indices_arr)[0] / itemsize_idx,
        indices_arr->ga.data,
        indices_arr->ga.offset,
        errbuf);

              if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: %(k_var)s: %%s.",
                             GpuKernel_error(&%(k_var)s, err));
                return 1;
              }
              err = gpudata_read(&kerr, errbuf, 0, sizeof(int));
              if (err != GA_NO_ERROR) {
                PyErr_SetString(PyExc_RuntimeError, "Can't read error buffer");
                return 1;
              }
              if (kerr != 0) {
                PyErr_SetString(PyExc_IndexError, "Index out of bounds");
                kerr = 0;
                gpudata_write(errbuf, 0, &kerr, sizeof(int));
                return 1;
              }
            }
          return 0;
        }
        """ % dict(k_var="k_vector_select_fast_" + nodename)
