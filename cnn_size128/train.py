import os
from glob import glob
import pickle
from dataset import DataSet
from model import CNNModel
import pandas as pd
from config import config

def init_train_dir(train_dir):
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)


def train_model(work_dir, initial_epoch = 0, batch_size = 16, initial_lr = 0.001, model_name = 'cnn', num_gpus = 3):
    model_path = None
    if initial_epoch > 0:
        model_paths = glob(work_dir + '%s_e%02d*.hd5' % (model_name, initial_epoch))
        if len(model_paths) != 1:
            print('cannot found model save file!!!!')
            assert(False)
        else:
            model_path = model_paths[0]
        initial_lr = initial_lr * (config['train']['lr_decay'] ** (initial_epoch - 1))
    else:
        initial_epoch = 0
    dataset = DataSet(data_dir = config['data']['data_dir'], train = True, train_pkl_data_info_path = "D:\\data\\tc\\pkl_data\\train_info.pkl", eval_pkl_data_info_path = "D:\\data\\tc\\pkl_data\\eval_info.pkl")
    cnn_model = CNNModel(model_path, True, initial_lr, num_gpus)

    train_data_gen = dataset.get_train_data_gen(batch_size = batch_size)
    train_data_num = dataset.get_train_data_num()
    eval_data_gen = dataset.get_eval_data_gen(batch_size = batch_size)
    eval_data_num =  dataset.get_eval_data_num()
    cnn_model.train_model(train_data_gen, train_data_num, eval_data_gen, eval_data_num, batch_size, work_dir, model_name = model_name, initial_epoch = 0)

if __name__ == '__main__':
    if False:
        data_set = DataSet("D:\\data\\tc\\", True, 'D:\\data\\天池\\') #将原始数据规范化转存数据到"D:\\data\\tc"目录，第一个参数目录下需要有原始数据的三个解压文件夹（*train_part*）
        data_set.split_data(eval_rate = 0.1, test_rate = 0.1)   #采样样本点，并划分训练集、验证集、测试集
        data_set.pre_pkl_data("D:\\data\\tc\\pkl_data\\") #将数据打包至"D:\\data\\tc\\pkl_data\\"目录
    if True:
        init_train_dir('D:\\data\\tc_train\\') # 初始化训练目录
        batch_size = 96
        train_model('D:\\data\\tc_train\\', initial_epoch = 0, batch_size = batch_size, initial_lr = 0.01)
