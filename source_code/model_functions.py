# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 22:51:19 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
import logging

import tensorflow
if tensorflow.__version__ >= '2.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow import keras
    import tensorflow.compat.v1.keras.backend as K
else:
    import tensorflow as tf
    import keras
    from keras import backend as K
from keras.layers import Input
from keras.models import Model


def get_outputs_of(model, start_tensors, input_layers=None):
    """ start_tensors为开始拆开的位置
        苏剑林. (Sep. 29, 2019). 《'让Keras更酷一些！'：层与模型的重用技巧 》[Blog post]. Retrieved from
        https://kexue.fm/archives/6958
    """
    # 为此操作建立新模型
    model = Model(inputs=model.input,
                  outputs=model.output,
                  name='outputs_of_' + model.name)
    # 适配工作，方便使用
    if not isinstance(start_tensors, list):
        start_tensors = [start_tensors]
    if input_layers is None:
        input_layers = [
            Input(shape=K.int_shape(x)[1:], dtype=K.dtype(x))
            for x in start_tensors
        ]
    elif not isinstance(input_layers, list):
        input_layers = [input_layers]
    # 核心：覆盖模型的输入
    model.inputs = start_tensors
    model._input_layers = [x._keras_history[0] for x in input_layers]
    # 适配工作，方便使用
    if len(input_layers) == 1:
        input_layers = input_layers[0]
    # 整理层，参考自 Model 的 run_internal_graph 函数
    layers, tensor_map = [], set()
    for x in model.inputs:
        tensor_map.add(str(id(x)))
    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
        nodes = model._nodes_by_depth[depth]
        for node in nodes:
            n = 0
            for x in node.input_tensors:
                if str(id(x)) in tensor_map:
                    n += 1
            if n == len(node.input_tensors):
                if node.outbound_layer not in layers:
                    layers.append(node.outbound_layer)
                for x in node.output_tensors:
                    tensor_map.add(str(id(x)))
    model._layers = layers  # 只保留用到的层
    # 计算输出
    outputs = model(input_layers)
    return input_layers, outputs


def part_front_model(model, layer_names):
    """Get front part of model from a keras model.
    Args:
        model (keras.Model): a full keras model.
        layer_names (list[str]): output layer_names.
    Returns:
        model_front (keras.Model): a front part of keras model.
    """
    if not isinstance(layer_names, list):
        layer_names = [layer_names]
    model_front = Model(inputs=model.input,
                        outputs=[model.get_layer(name).output for name in layer_names])
    return model_front


def part_back_model_(model, layer_names):
    """Get back part of model from a keras model.
    NOT working !!!
    Args:
        model (keras.Model): a full keras model.
        layer_names (list[str]): input layer_names.
    Returns:
        model_back (keras.Model): a back part of keras model.
    """
    if isinstance(layer_names, list):
        model_back = Model(inputs=[model.get_layer(name).input for name in layer_names],
                           outputs=model.output)
    else:
        model_back = Model(inputs=model.get_layer(layer_names).output,
                           outputs=model.output)
    return model_back


def part_back_model(model, start_tensors, input_layers=None):
    """ Get back part of a model, the return model only has one layer, and cannot load correctly.
        苏剑林. (Sep. 29, 2019). 《'让Keras更酷一些！'：层与模型的重用技巧 》[Blog post]. Retrieved from
        https://kexue.fm/archives/6958
    Args:
        start_tensors (keras.layer.output): output of the layer before back model, where to split model.
        input_layers (keras.layer.Input): input layer of the back model.
    """
    # build a new model
    model = Model(inputs=model.input,
                  outputs=model.output)
    # standard paras
    if not isinstance(start_tensors, list):
        start_tensors = [start_tensors]
    if input_layers is None:
        input_layers = [
            Input(shape=K.int_shape(x)[1:], dtype=K.dtype(x))
            for x in start_tensors
        ]
    elif not isinstance(input_layers, list):
        input_layers = [input_layers]
    # replace the inputs of the model
    model.inputs = start_tensors
    model._input_layers = [x._keras_history[0] for x in input_layers]

    if len(input_layers) == 1:
        input_layers = input_layers[0]
    # arrange layers, reference from function run_internal_graph of Model
    layers, tensor_map = [], set()
    for x in model.inputs:
        tensor_map.add(str(id(x)))
    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
        nodes = model._nodes_by_depth[depth]
        for node in nodes:
            n = 0
            for x in node.input_tensors:
                if str(id(x)) in tensor_map:
                    n += 1
            if n == len(node.input_tensors):
                if node.outbound_layer not in layers:
                    layers.append(node.outbound_layer)
                for x in node.output_tensors:
                    tensor_map.add(str(id(x)))
    model._layers = layers  # keep usefull layers

    outputs = model(input_layers)
    model_back = Model(input_layers, outputs)

    return model_back


def part_layer_model(model, n_input_layer, inputs, n_output_layer):
    """Get part of model from a keras model through layer number, only used for sequential model.
    Args:
        model (keras.Model): a full keras model.
        n_input_layer (int): index of the input layer of the part model.
        inputs (Keras.layers.Input): input vector of the part model.
        n_output_layer (int): index of the output layer of the part model.
    Returns:
        model_part (keras.Model): a part of keras model.
    """
    model_layers = [model.layers[i] for i in range(n_input_layer, n_output_layer)]
    x = inputs
    for model_layer in model_layers:
        x = model_layer(x)
    outputs = x

    part_model = Model(inputs, outputs)
    return part_model
