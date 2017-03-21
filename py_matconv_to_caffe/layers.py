from caffe import layers as L
from caffe import params as P
import caffe
import numpy as np


def layer_factory(bottom, mcn_layer, mcn_layer_params):
    fun = globals().get('_%s' % mcn_layer.type.replace('.', '_'))
    if fun is None:
        raise ValueError('%s layer not implemented' % mcn_layer.type)
    return fun(bottom, mcn_layer, mcn_layer_params)


def _dagnn_Conv(bottom, mcn_layer, mcn_layer_params):
    block = mcn_layer.block
    params = _prepare_input_params(mcn_layer_params)
    params['num_output'] = block.size[-1]
    params = _wrap_params(params, block.size, block.pad, block.stride)
    return L.Convolution(*bottom, **params)


def _dagnn_ReLU(bottom, mcn_layer, mcn_layer_params):
    if mcn_layer.block.leak != 0:
        raise ValueError('mcn_layer.block.leak is non zero. Do not know how to initialize Caffe Layer.')
    return L.ReLU(*bottom, in_place=True)


def _dagnn_Pooling(bottom, mcn_layer, mcn_layer_params):
    params = _prepare_input_params(mcn_layer_params)
    block = mcn_layer.block
    if block.method == 'max':
        params['pool'] = P.Pooling.MAX
    elif block.method == 'avg':
        params['pool'] = P.Pooling.AVE
    else:
        raise ValueError('Pooling type: %s not implemented.' % block.method)
    params = _wrap_params(params, block.poolSize, block.pad, block.stride)
    if len(block.opts) > 0 and not (block.opts == 'cuDNN'):
        raise ValueError('mcn_layer.block has opts record which I did not expect. %s' % block.opts)
    return L.Pooling(*bottom, **params)


def _dagnn_DropOut(bottom, mcn_layer, mcn_layer_params):
    dr = mcn_layer.block.rate
    if mcn_layer.block.frozen != 0:
        raise ValueError('I do not know what to do with mcn_layer.block.frozen != parameter.')
    return L.Dropout(*bottom, dropout_param={'dropout_ratio': dr})


def _dagnn_Loss(bottom, mcn_layer, mcn_layer_params):
    loss_type = mcn_layer.block.loss
    if loss_type == 'softmaxlog':
        if len(bottom) > 1:
            bottom = bottom[0]
        return L.Softmax(bottom)
    elif loss_type == 'classerror':
        return L.Accuracy(*bottom, top_k=1)
    elif loss_type == 'topkerror':
        if len(mcn_layer.block.opts) > 0 and mcn_layer.block.opts[0] != 'topK':
            raise ValueError('Unknown opt %s in classerror layer.' % mcn_layer.block.opts[0])
        top_k = int(mcn_layer.block.opts[1])
        return L.Accuracy(*bottom, top_k=top_k)
    else:
        raise ValueError('Unknown loss type')


def _dagnn_BatchNorm(bottom, mcn_layer, mcn_layer_params):
    params = _prepare_input_params(mcn_layer_params)
    params['eps'] = mcn_layer.block.epsilon
    params['use_global_stats'] = True
    return L.BatchNorm(*bottom, **params)


def _dagnn_Sum(bottom, mcn_layer, mcn_layer_params):
    return L.Eltwise(*bottom, eltwise_param={'operation': P.Eltwise.SUM})


def _wrap_params(params_dict, mcn_kernelsize, mcn_pad, mcn_stride):
    """
    In matconvnet each parameter which has param wrt heigh and width is saved as separate
    variable. For instance kernel_h and kernel_w. But in caffe if kernel_h and kernel_w
    are same, we can wrap them to one parameter kernel = kernel_h = kernel_w. This function
    wraps all *_h *_w prameters into one, for kernel, pad and strinde parameters.

    :param params_dict: Dictionary with caffe parametes. This function will add kernel, pad and stride
     or kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w key to this dictionary.
    :param mcn_kernelsize: list of parameters for kernel size from matconvnet. Eg. [2,2] for kernel size 2x2.
    :param mcn_pad: list of parameters for pad size from matconvnet.
    :param mcn_stride: list of parameters for stride size from matconvnet.
    :return: Updated dictionary.
    """
    if mcn_kernelsize[0] == mcn_kernelsize[1]:
        params_dict['kernel_size'] = mcn_kernelsize[0]
    else:
        params_dict['kernel_h'], params_dict['kernel_w'] = mcn_stride[:2]

    if mcn_stride[0] == mcn_stride[1]:
        params_dict['stride'] = mcn_stride[0]
    else:
        params_dict['stride_h'], params_dict['stride_w'] = mcn_stride
    pad = mcn_pad[[0, 2]]
    if pad[0] == pad[1]:
        params_dict['pad'] = pad[0]
    else:
        params_dict['pad_h'], params_dict['pad_w'] = pad
    return params_dict


def _prepare_input_params(mcn_layer_params):
    """
    Prepares params dictionary template based on parameters obtained from
    matconvnet mcn_layer_params are params obtianed from net.params(layer_name).
    The params in mcn_layer_prams are in general learining rate, and weight decay.
    But there can be more than one group eg. separate lr and wd for weights and form bias.
    To be compatible with caffe we need to return empty dict if there are no params (eg. for ReLu layer),
    dict of structure {'param' : dict}, when there is one group, and finally dict of structure
    {'param': list} when there is more than one group.

    :param mcn_layer_params:
    :return:
    """
    if len(mcn_layer_params) == 0:
        return {}
    if len(mcn_layer_params) == 1:
        params = {'param': mcn_layer_params[0]}
    else:
        params = {'param': mcn_layer_params}
    return params
