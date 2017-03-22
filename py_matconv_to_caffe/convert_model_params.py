from caffe import layers as L
import caffe
from layers import layer_factory
from utils import get_values_for_multi_keys
from collections import defaultdict
from collections import deque

"""
This module implements functionality which will convert architecture of the net from matconvnet to
caffe. Including all parameters - number of layers, type of layers, layers params (eg. kernel size,
pad size, stride step etc.). This module also transfers training parameters eg. learning rate for given
weights and weight decay.

Please note that in whole project term "parameters" refers to the setup of network. While term 'weights' refers
to trained/learned model parameters which includes weights of connections between layers and filter weights (in CNN
layers).
"""


def convert_net(matconv_net):
    data_shape = matconv_net.meta.normalization.imageSize[:3]
    data_shape = [1, data_shape[-1], data_shape[0], data_shape[1]]
    data = L.DummyData(shape=dict(dim=data_shape))
    label = L.DummyData(shape=dict(dim=[1]))
    n = caffe.NetSpec()
    n.data = data
    n.label = label

    # dictionary with name of output variable of given layer as a key and layer as a value
    layers_dic = {'input': data, 'label': label}
    # dictionary with layer name as key and dict as a value. Dict contains lr_mult and decay_mult
    # which values are obtained from matconv_net.params
    lr_params_dic = _create_lr_params_dic(matconv_net.params)

    execute_order = _get_execution_order(matconv_net.layers, (0, matconv_net.layers[0]))
    for _, mcn_layer in execute_order:
        # searching for input layer based on input/output variable names
        bottom = get_values_for_multi_keys(layers_dic, mcn_layer.inputs)
        layer_name = mcn_layer.name
        top = layer_factory(bottom, mcn_layer, get_values_for_multi_keys(lr_params_dic, mcn_layer.params))
        # add just created layer to dict. Key value is a matconvnet output variable name
        layers_dic[mcn_layer.outputs] = top[-1]
        for i, t in enumerate(top[:-1]):
            n.__setattr__('%s_INTER%02d' % (layer_name, i), t)
        n.__setattr__(layer_name, top[-1])
    return n


def _get_execution_order(layers, start):
    # dictionary with input name as a key and (layer_id, layer) as a value
    layers_by_inputs = defaultdict(list)
    for layer_id, l in enumerate(layers):
        for i in _listify(l.inputs):
            layers_by_inputs[i].append((layer_id, l))
    # depth first search stack
    queue = deque()
    queue.append(start)
    # dctionary which contains name of output variables of already visited layers
    visited = {}
    while len(queue) > 0:
        curr = queue.pop()
        visited[curr[1].outputs] = True
        for next_layer_id, next_layer in layers_by_inputs[curr[1].outputs]:
            # it checks if all inputs to given layer has been visited. If no then this layer is not added
            # to the stack. It will be added when last output which was not ready will be visited.
            is_ready = None not in [visited.get(inp) for inp in _listify(next_layer.inputs)]
            if is_ready:
                queue.append((next_layer_id, next_layer))
        yield curr


def _create_lr_params_dic(matconv_params_list):
    lr_params_dic = {}
    for mcn_lr_params in matconv_params_list:
        lr_params_dic[mcn_lr_params.name] = {'lr_mult': mcn_lr_params.learningRate,
                                             'decay_mult': mcn_lr_params.weightDecay}
    return lr_params_dic


def _listify(arg):
    if isinstance(arg, str) or isinstance(arg, unicode):
        return [arg]
    else:
        return arg
