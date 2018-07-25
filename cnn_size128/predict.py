import os
from glob import glob
import pickle
from dataset import DataSet
from model import CNNModel
import pandas as pd
from config import config
import sklearn.metrics as skl_metrics
import matplotlib.pyplot as plt
import numpy as np
import cv2

def cal_roc_and_auc(pred_array, label_array):
    total = len(label_array)
    P = sum(label_array)
    N = (total - P)
    roc_y = []
    roc_x = []
    for threshold in np.arange(1.0, -0.001, -0.001):
        tp_list = np.less_equal(threshold, pred_array) * label_array
        fp_list = np.less_equal(threshold, pred_array) * (1 - label_array)
        tpr = sum(tp_list) / P if P != 0 else 0.0
        fpr = sum(fp_list) / N if N != 0 else 0.0
        roc_x.append(fpr)
        roc_y.append(tpr)
    x = roc_x[0]
    y = roc_y[0]
    auc = 0.0
    for cur_x, cur_y in zip(roc_x,roc_y):
        auc += (cur_y + y) * (cur_x - x) / 2
        x = cur_x
        y = cur_y
    return auc

def draw_roc(pred_array, label_array, target_dir):
    fprs, tprs, _ = skl_metrics.roc_curve(label_array, pred_array)
    plt.plot(fprs, tprs)
    plt.title('roc')
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()
    plt.savefig(os.path.join(target_dir, "roc.png"))
    plt.close()

def predict_test_set(model_path, batch_size, pkl_data_dir, target_dir, num_gpus = 1):
    dataset = DataSet(data_dir = config['data']['data_dir'], test = True, pkl_file_dir = pkl_data_dir)
    cnn_model = CNNModel(model_path, False, gpu_nums = num_gpus)
    test_data_gen = dataset.get_test_data_gen(batch_size)
    pred_list = []
    label_list = []
    fp_dict = {'file_path': [], 'pkl_idx': [], 'p': []}
    fn_dict = {'file_path': [], 'pkl_idx': [], 'p': []}
    for batch_data, batch_label, batch_info in test_data_gen:
        preds = cnn_model.predict(batch_data['inputs'], batch_size)
        idx = 0
        for pred in preds:
            l = batch_label['out_class'][idx][1]
            p = pred[1]
            label_list.append(l)
            pred_list.append(p)
            if p >= 0.5 and l != 1:
                fp_dict['file_path'].append(batch_info['file_path'][idx])
                fp_dict['pkl_idx'].append(batch_info['pkl_idx'][idx])
                fp_dict['p'].append(p)
            if p < 0.5 and l != 0:
                fn_dict['file_path'].append(batch_info['file_path'][idx])
                fn_dict['pkl_idx'].append(batch_info['pkl_idx'][idx])
                fn_dict['p'].append(p)
            idx += 1
    auc = cal_roc_and_auc(np.array(pred_list), np.array(label_list))
    fn_df = pd.DataFrame(fn_dict)
    fp_df = pd.DataFrame(fp_dict)
    fn_df.to_csv(os.path.join(target_dir, 'fn_result.csv'), index = False)
    fp_df.to_csv(os.path.join(target_dir, 'fp_result.csv'), index = False)
    draw_roc(np.array(pred_list), np.array(label_list), target_dir)
    print('auc:%f' % auc)


def gen_result(model_path, test_img_dir, target_dir, batch_size = 64, num_gpus = 1):
    cnn_model = CNNModel(model_path, False, gpu_nums = num_gpus)
    result_dict = {'filename': [], 'probability': []}
    for img_path in glob(os.path.join(test_img_dir, '*.jpg')):
        print(img_path)
        img_pred = cnn_model.predict_one_img(img_path, batch_size)
        result_dict['filename'].append(os.path.split(img_path)[-1])
        result_dict['probability'].append(img_pred if img_pred < 1 else 0.999)
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(os.path.join(target_dir, 'result.csv'), index = False)

def clip_result(result_path):
    result_df = pd.read_csv(result_path)
    result_df.loc[result_df['probability'] >= 1, 'probability'] = 0.999
    result_df.to_csv(result_path, index = False)

def save_error_data(target_dir, csv_path):
    result_df = pd.read_csv(csv_path)
    result_df = result_df.sort_values(by = ['file_path'])
    pkl_path = None
    pkl_file = None
    img_idx = 0
    for _, row in result_df.iterrows():
        if pkl_path == None or row['file_path'] != pkl_path:
            pkl_path = row['file_path']
            with open(pkl_path, 'rb') as fp:
                pkl_file = pickle.load(fp)
        cv2.imwrite(os.path.join(target_dir, 'img_%d_%f.png' % (img_idx, row['p'])), pkl_file[row['pkl_idx']]['data'])


if __name__ == '__main__':
    predict_test_set('/mnt/data1/lty/train/cnn_e07-0.6411.hd5', 32, '/mnt/data1/lty/pkl_data/', '/mnt/data1/lty/test/')
    save_error_data('/mnt/data1/lty/test/fn_data/', '/mnt/data1/lty/test/fn_.csv')
    save_error_data('/mnt/data1/lty/test/fp_data/', '/mnt/data1/lty/test/fp_.csv')
    # gen_result('/mnt/data1/lty/train/cnn_e07-0.6411.hd5', '/mnt/data1/lty/xuelang_round1_test_a_20180709/', '/mnt/data1/lty/', 32)
    # clip_result('/mnt/data1/lty/result.csv')
