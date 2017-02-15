from caffe import layers as L
from caffe import params as P
import caffe


def process(matconv_net):
    data = L.DummyData(shape=dict(dim=[1, 3, 224, 224]))
    label = L.DummyData(shape=dict(dim=[1, 101]))
    n = caffe.NetSpec()
    n.data = data
    n.label = label
    layers_dic = {'input': data, 'label': label}
    lr_params_dic = create_lr_params_dic(matconv_net['net'].params)
    for mcn_layer in matconv_net['net'].layers:
        bottom = get_multi_keys(layers_dic, mcn_layer.inputs)
        layer_name = mcn_layer.name
        top = layer_factory(bottom, n.label, mcn_layer, get_multi_keys(lr_params_dic, mcn_layer.params))
        layers_dic[mcn_layer.outputs] = top
        n.__setattr__(layer_name, top)
    return n


def create_lr_params_dic(matconv_params_list):
    lr_params_dic = {}
    for mcn_lr_params in matconv_params_list:
        lr_params_dic[mcn_lr_params.name] = {'lr_mult': mcn_lr_params.learningRate,
                                             'decay_mult': mcn_lr_params.weightDecay}
    return lr_params_dic


def layer_factory(bottom, label, mcn_layer, mcn_layer_params):
    fun = globals().get(mcn_layer.type.replace('.', '_'))
    return fun(bottom, label, mcn_layer, mcn_layer_params)


def get_multi_keys(dictionary, keys):
    if not hasattr(keys, '__getitem__') or isinstance(keys, basestring):
        keys = [keys]
    return [dictionary[k] for k in keys]


def prepare_input_params(mcn_layer_params):
    if len(mcn_layer_params) == 0:
        return {}
    if len(mcn_layer_params) == 1:
        params = {'param': mcn_layer_params[0]}
    else:
        params = {'param': mcn_layer_params}
    return params


def wrap_params(params_dict, mcn_kernelsize, mcn_pad, mcn_stride):
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

def dagnn_Conv(bottom, label, mcn_layer, mcn_layer_params):
    block = mcn_layer.block
    params = prepare_input_params(mcn_layer_params)
    #     params = {'param': np.zeros((3,3,3,64))}
    params['num_output'] = block.size[-1]
    params = wrap_params(params, block.size, block.pad, block.stride)
    return L.Convolution(*bottom, **params)


def dagnn_ReLU(bottom, label, mcn_layer, mcn_layer_params):
    if mcn_layer.block.leak != 0:
        raise ValueError('mcn_layer.block.leak is non zero. Do not know how to initialize Caffe Layer.')
    return L.ReLU(*bottom, in_place=True)


def dagnn_Pooling(bottom, label, mcn_layer, mcn_layer_params):
    params = prepare_input_params(mcn_layer_params)
    block = mcn_layer.block
    params['pool'] = P.Pooling.MAX if block.method == 'max' else P.Pooling.MIN
    params = wrap_params(params, block.poolSize, block.pad, block.stride)
    if len(block.opts) > 0:
        raise ValueError('mcn_layer.block has opts record which I did not expect.')
    return L.Pooling(*bottom, **params)


def dagnn_DropOut(bottom, label, mcn_layer, mcn_layer_params):
    dr = mcn_layer.block.rate
    if mcn_layer.block.frozen != 0:
        raise ValueError('I do not know what to do with mcn_layer.block.frozen != parameter.')
    return L.DropOut(*bottom, dropout_param={'dropout_ratio': dr})


def dagnn_Loss(bottom, label, mcn_layer, mcn_layer_params):
    loss_type = mcn_layer.block.loss
    if loss_type == 'softmaxlog':
        return L.Softmax(*bottom)
    elif loss_type == 'classerror':
        return L.Accuracy(*bottom, top_k=1)
    elif loss_type == 'topkerror':
        if len(mcn_layer.block.opts) > 0 and mcn_layer.block.opts[0] != 'topK':
            raise ValueError('Unknown opt %s in classerror layer.' % mcn_layer.block.opts[0])
        top_k = int(mcn_layer.block.opts[1])
        return L.Accuracy(*bottom, top_k=top_k)
    else:
        raise ValueError('Unknown loss type')
