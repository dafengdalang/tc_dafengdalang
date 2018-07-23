#该文件：提供nodule、features模型训练的多GPU使用
from __future__ import print_function
from keras.models import *
from keras.layers import Input, Lambda
from keras.layers.merge import Concatenate
from keras import backend as K
import os

# '0,1,2,3'  '3,4,5,6' '6,7,8,9' '0,1,2' '3,4,5' '6,7,8'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5' #=>availabel devices  '0,1,2,4,5,6,7,8,9' '5,6,7,8,9' '0,1,2,3,4' '5,6,7,8'

import tensorflow as tf
# session_config = tf.ConfigProto()
# session_config.gpu_options.allow_growth = True
# session_config.gpu_options.per_process_gpu_memory_fraction = 0.2 # 0.2*
# session_config.allow_soft_placement = True
# session = tf.Session(config=session_config)


USE_FREE_MEMORY_RATIO=1.0 #使用可用内存的多少比例
# 自动获取可用gpu
# input_ngpu: int类型, 表示需要的gpu数量(为None时表示使用全部可用gpu)
# free_ratio: float类型, 表示选择的gpu必须满足可用内存占比,默认有50%以上的可用内存才是可用gpu
# exclude_gpus: 需要排除的gpus
# 使用方法: 设置需要使用的gpu数量, 需要的gpu空闲内存所占的比例; 可设置free_ration,这样只选择没人用的gpu
# eg: gpus, ratio=get_available_gpus(input_ngpu=4, free_ratio=0.5, orderby_free_memory=True)
def get_available_gpus(input_ngpu=None, free_ratio=0.5, orderby_free_memory=False, exclude_gpus=[]):
    params = os.popen("nvidia-smi --query-gpu=index,pci.bus_id,gpu_name,memory.used,memory.free,memory.total,power.draw,power.limit --format=csv ")  # > gpu1.txt
    gpus_info = params.readlines()  # 不能重复读,第一行是列名

    available=[]
    available_used=[]   #可用gpu按free memory排序,然后选取最空闲的gpu
    for index in range(1, len(gpus_info)):
        gpu_info = gpus_info[index].split(",")
        gpu_id = gpu_info[0]
        memory_free = float(gpu_info[4].strip().split(" ")[0])
        memory_total = float(gpu_info[5].strip().split(" ")[0])
        memory_free_ratio = memory_free / memory_total
        if memory_free_ratio>=free_ratio-0.05:  #有可能显存会轻微地波动
                available_used.append(memory_free_ratio)
                available.append(int(gpu_id))
                # min_memory_free_ratio=min(min_memory_free_ratio, memory_free_ratio)

    if input_ngpu is not None:
        if len(available)<input_ngpu:
            gpus_num=len(available)
            print("you want to use %d gpus, but only %d gpus available." %(input_ngpu, len(available)))
        else:
            gpus_num=input_ngpu

    # 按名称对可用gpu排序
    available=[str(val) for val in available]
    available.sort()

    # 排除指定的不可用的gpus
    if len(exclude_gpus) > 0:
        available_used = [available_used[idx] for idx in range(len(available)) if available[idx] not in exclude_gpus]
        available = [val for val in available if val not in exclude_gpus]

    all_gpus=",".join(available)

    if orderby_free_memory:
        available = np.array(available)[np.array(available_used).argsort()[::-1]]  # np.argsort只能返回按从小到大排的索引
    gpus=",".join(sorted(available[:gpus_num]))

    if gpus_num:
        print("all available gpus: ", all_gpus)
        print("you will use gpus: ", gpus)
    return gpus, (free_ratio-0.05)*USE_FREE_MEMORY_RATIO    #返回原gpu可用百分比*0.6为GPU内存使用比

# 人为指定使用那几块GPU
# eg: hand_set_gpus(gpus="0,1,2", memory_ratio=0.8, auto_growth=False)
def hand_set_gpus(gpus="", memory_ratio=0.8, auto_growth=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus  # =>availabel devices: 0,1,2,3,4,5

    session_config = tf.ConfigProto()
    if auto_growth:
        session_config.gpu_options.allow_growth = True
    else:
        session_config.gpu_options.per_process_gpu_memory_fraction = memory_ratio
    session_config.allow_soft_placement = True
    session = tf.Session(config=session_config)
    return session


# 智能化指定可用GPU(独立成函数,避免出现覆盖的情况)
#eg: set_gpus(num_gpus=2, free_ratio=1.0, auto_growth=False)
def set_gpus(num_gpus, free_ratio=1.0, auto_growth=False, exclude_gpus=[]):
    gpus, ratio = get_available_gpus(input_ngpu=num_gpus, free_ratio=free_ratio, orderby_free_memory=True, exclude_gpus=exclude_gpus)

    session = hand_set_gpus(gpus=gpus, memory_ratio=ratio, auto_growth=auto_growth)
    return gpus, session

def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part * L:]
    return x[part * L:(part + 1) * L]

# nodule模型训练的多GPU使用
def to_multi_gpu_nodule(model, n_gpus=2):
    if n_gpus == 1:
        return model

    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:])
        # print("model.input_shape: ", model.input_shape, model.input_shape[1:])
        # print("x.shape: ", x.shape)

    towers1 = []
    towers2 = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus': n_gpus, 'part': g})(x)
            # towers.append(model(slice_g))
            output = model(slice_g)
            towers1.append(output[0])  # the output of model is [tensor1, tensor2]
            towers2.append(output[1])

    with tf.device('/cpu:0'):
        # Deprecated
        # merged = merge(towers, mode='concat', concat_axis=0)
        merged1 = Concatenate(axis=0, name="out_class")(towers1)
        print(merged1.shape)

        merged2 = Concatenate(axis=0, name="out_malignancy")(towers2)
    return Model(inputs=x, outputs=[merged1, merged2])


# 提供features模型训练的多GPU使用
def to_multi_gpu_features(model, n_gpus=2):
    if n_gpus == 1:
        return model

    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:])

    towers1 = []
    towers2 = []
    towers3 = []
    towers4 = []
    towers5 = []
    towers6 = []
    towers7 = []
    towers8 = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus': n_gpus, 'part': g})(x)
            output = model(slice_g)
            towers1.append(output[0])  # the output of model is [tensor1, tensor2]
            towers2.append(output[1])
            towers3.append(output[2])
            towers4.append(output[3])
            towers5.append(output[4])
            towers6.append(output[5])
            towers7.append(output[6])
            towers8.append(output[7])

    with tf.device('/cpu:0'):
        merged1 = Concatenate(axis=0, name="out_diameter")(towers1)
        merged2 = Concatenate(axis=0, name="out_malscore")(towers2)
        merged3 = Concatenate(axis=0, name="out_sphericiy")(towers3)
        merged4 = Concatenate(axis=0, name="out_margin")(towers4)
        merged5 = Concatenate(axis=0, name="out_spiculation")(towers5)
        merged6 = Concatenate(axis=0, name="out_texture")(towers6)
        merged7 = Concatenate(axis=0, name="out_lobulation")(towers7)
        merged8 = Concatenate(axis=0, name="out_subtlety")(towers8)
    return Model(inputs=x, outputs=[merged1, merged2, merged3, merged4, merged5, merged6, merged7, merged8])


def to_multi_gpu_nodule_segmented(model, n_gpus = 2):
    if n_gpus == 1:
        return model

    with tf.device('/cpu:0'):
        # print("model.input_shape: ", model.input_shape)
        # input()
        inputs = Input(model.input_shape[1:], name="inputs")
        # print("model.input_shape: ", model.input_shape, model.input_shape[1:])
        # print("x.shape: ", x.shape)

    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_inputs = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus': n_gpus, 'part': g})(inputs)
            output = model(slice_inputs)
            towers.append(output) # the output of model is [tensor1, tensor2]

    with tf.device('/cpu:0'):
        # Deprecated
        merged = Concatenate(axis=0, name="conv_out")(towers)

    return Model(inputs = inputs, outputs=[merged])