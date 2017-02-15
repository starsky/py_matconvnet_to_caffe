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
    for mcn_layer in matconv_net['net'].layers:
        bottom = get_multi_keys(layers_dic, mcn_layer.inputs)
        layer_name, top = layer_factory(bottom, n.label, mcn_layer)
        layers_dic[mcn_layer.outputs] = top
        n.__setattr__(layer_name, top)
    return n


def layer_factory(bottom, label, mcn_layer):
    fun = globals().get(mcn_layer.type.replace('.', '_'))
    layer_name = mcn_layer.name
    return layer_name, fun(bottom, label, mcn_layer)


def get_multi_keys(dictionary, keys):
    if not hasattr(keys, '__getitem__') or isinstance(keys, basestring):
        keys = [keys]
    return [dictionary[k] for k in keys]


def dagnn_Conv(bottom, label, mcn_layer):
    block = mcn_layer.block
    params = {'param': {'lr_mult': 1, 'decay_mult': 1}}
    #     params = {'param': np.zeros((3,3,3,64))}
    if block.size[0] == block.size[1]:
        params['kernel_size'] = block.size[0]
    else:
        params['kernel_h'], params['kernel_w'] = block.size[:2]

    if block.stride[0] == block.stride[1]:
        params['stride'] = block.stride[0]
    else:
        params['stride_h'], params['stride_w'] = block.stride
    params['num_output'] = block.size[-1]
    pad = block.pad[[0, 2]]
    if pad[0] == pad[1]:
        params['pad'] = pad[0]
    else:
        params['pad_h'], params['pad_w'] = pad
    return L.Convolution(*bottom, **params)


def dagnn_ReLU(bottom, label, mcn_layer):
    if mcn_layer.block.leak != 0:
        raise ValueError('mcn_layer.block.leak is non zero. Do not know how to initialize Caffe Layer.')
    return L.ReLU(*bottom, in_place=True)


def dagnn_Pooling(bottom, label, mcn_layer):
    block = mcn_layer.block
    method = P.Pooling.MAX if block.method == 'max' else P.Pooling.MIN
    kernel_h, kernel_w = block.poolSize
    stride_h, stride_w = block.stride
    pad_h, pad_w = block.pad[[0, 2]]
    if len(block.opts) > 0:
        raise ValueError('mcn_layer.block has opts record which I did not expect.')
    return L.Pooling(*bottom, pool=P.Pooling.MAX, kernel_h=kernel_h, kernel_w=kernel_w, stride_h=stride_h,
                     stride_w=stride_w, pad_h=pad_h, pad_w=pad_w)


def dagnn_DropOut(bottom, label, mcn_layer):
    dr = mcn_layer.block.rate
    if mcn_layer.block.frozen != 0:
        raise ValueError('I do not know what to do with mcn_layer.block.frozen != parameter.')
    return L.DropOut(*bottom, dropout_param={'dropout_ratio': dr})


def dagnn_Loss(bottom, label, mcn_layer):
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
