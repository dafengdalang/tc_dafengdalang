import numpy as np
import cv2
import os
from glob import glob
import xml.etree.ElementTree as ET
import shutil
import pickle
import pandas as pd

def get_object_rect(tree_root):
    '''
    从xml中找到位置信息,
    min_x, min_y, max_x, max_y
    '''
    rects = []
    for object_elem in tree_root.iterfind('object'):
        for coord_elem in object_elem.iterfind('bndbox'):
            rects.append([int(coord_elem[0].text), int(coord_elem[1].text), int(coord_elem[2].text), int(coord_elem[3].text)])
    return rects

def get_img_cubic(img_array, center_xy, size = 256):
    size_y, size_x, channel = img_array.shape
    assert(channel == 3)
    cubic = np.zeros([size, size, 3], np.uint8)
    start_xy = [max(0, center_xy[0] - size // 2), max(0, center_xy[1] - size // 2)]
    end_xy = [min(size_x, center_xy[0] + (size + 1) // 2), min(size_y, center_xy[1] + (size + 1) // 2)]
    shape_xy = [end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]]
    offset_xy = [max(size // 2 - center_xy[0] + start_xy[0], 0), max(size // 2 - center_xy[1] + start_xy[1], 0)]
    cubic[offset_xy[1] : offset_xy[1] + shape_xy[1], offset_xy[0] : offset_xy[0] + shape_xy[0], :] = img_array[start_xy[1] : end_xy[1], start_xy[0] : end_xy[0], :]
    return cubic

class DataSet(object):
    def __init__(self, data_dir, init = False, ori_data_dir = None, train = False, test = False, pkl_file_dir = None):
        '''
        init 为True时需要提供ori_data_dir(原始数据目录),将从原始数据目录中统计分类信息，并重新保存图片到规范化的data_dir
        用于训练时，需指定train为True，并且提供最后三个参数，最后三个路径的文件需要首先执行一次pre_pkl_data函数来生成
        用于测试时，需指定test为True
        '''
        self.data_dir = data_dir
        self.train = train
        self.test = test
        self.sample_points_df = None
        if init:
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)
            self._prepare_data(ori_data_dir)
            with open(os.path.join(self.data_dir, 'data_info.pkl'), 'rb') as fp:
                self.data_info = pickle.load(fp)
        else:
            with open(os.path.join(self.data_dir, 'data_info.pkl'), 'rb') as fp:
                self.data_info = pickle.load(fp)
        self.train_sample_csv_path = os.path.join(self.data_dir, 'train_sample.csv')
        self.eval_sample_csv_path = os.path.join(self.data_dir, 'eval_sample.csv')
        self.test_sample_csv_path = os.path.join(self.data_dir, 'test_sample.csv')

        if self.train:
            self.pkl_file_dir = pkl_file_dir
            train_pkl_data_info_path = os.path.join(pkl_file_dir, 'train_info.pkl')
            eval_pkl_data_info_path = os.path.join(pkl_file_dir, 'eval_info.pkl')
            if train_pkl_data_info_path is not None and os.path.exists(train_pkl_data_info_path):
                with open(train_pkl_data_info_path, 'rb') as fp:
                    self.train_pkl_data_info = pickle.load(fp)
            else:
                print('cannot found train pkl data info file, run init and data prepare step first!!!!')
                assert(False)

            if eval_pkl_data_info_path is not None and os.path.exists(eval_pkl_data_info_path):
                with open(eval_pkl_data_info_path, 'rb') as fp:
                    self.eval_pkl_data_info = pickle.load(fp)
            else:
                print('cannot found eval pkl data info file, run init and data prepare step first!!!!')
                assert(False)
        if self.test:
            self.pkl_file_dir = pkl_file_dir
            test_pkl_data_info_path = os.path.join(pkl_file_dir, 'test_info.pkl')
            if test_pkl_data_info_path is not None and os.path.exists(test_pkl_data_info_path):
                with open(test_pkl_data_info_path, 'rb') as fp:
                    self.test_pkl_data_info = pickle.load(fp)
            else:
                print('cannot found test pkl data info file, run init and data prepare step first!!!!')
                assert(False)

    def _get_data_generator(self, pkl_data_infos, batch_size = 16, shuffle = False):
        batch_data = {'inputs': [], 'out_class': []}
        batch_data_num = 0
        while True:
            if shuffle:
                np.random.shuffle(pkl_data_infos)
            for pkl_data_info in pkl_data_infos:
                pkl_file_path = os.path.join(self.pkl_file_dir, pkl_data_info['file_name'])
                with open(pkl_file_path, 'rb') as fp:
                    pkl_file = pickle.load(fp)
                    if shuffle:
                        np.random.shuffle(pkl_file)
                    for record in pkl_file:
                        batch_data['inputs'].append(record['data'])
                        batch_data['out_class'].append([1, 0] if record['l'] == 0 else [0, 1])
                        batch_data_num += 1
                        if batch_data_num == batch_size:
                            yield {'inputs': np.array(batch_data['inputs'], np.float32) / 255}, {'out_class': np.array(batch_data['out_class'])}
                            batch_data = {'inputs': [], 'out_class': []}
                            batch_data_num = 0

    def get_train_data_gen(self, batch_size):
        return self._get_data_generator(self.train_pkl_data_info, batch_size, True)
    
    def get_eval_data_gen(self, batch_size):
        return self._get_data_generator(self.eval_pkl_data_info, batch_size, False)
    
    def get_test_data_gen(self, batch_size):
        batch_data = {'inputs': [], 'out_class': [], 'info': {'file_path': [], 'pkl_idx': []}}
        batch_data_num = 0
        for pkl_data_info in self.test_pkl_data_info:
            pkl_file_path = os.path.join(self.pkl_file_dir, pkl_data_info['file_name'])
            with open(pkl_file_path, 'rb') as fp:
                pkl_file = pickle.load(fp)
                pkl_idx = 0
                for record in pkl_file:
                    batch_data['inputs'].append(record['data'])
                    batch_data['out_class'].append([1, 0] if record['l'] == 0 else [0, 1])
                    batch_data_num += 1
                    batch_data['info']['file_path'].append(pkl_file_path)
                    batch_data['info']['pkl_idx'].append(pkl_idx)
                    pkl_idx += 1
                    if batch_data_num == batch_size:
                        yield {'inputs': np.array(batch_data['inputs'], np.float32) / 255}, {'out_class': np.array(batch_data['out_class'])}, batch_data['info']
                        batch_data = {'inputs': [], 'out_class': [], 'info': {'file_path': [], 'pkl_idx': []}}
                        batch_data_num = 0
        if batch_data_num:
            yield {'inputs': np.array(batch_data['inputs'], np.float32) / 255}, {'out_class': np.array(batch_data['out_class'])}, batch_data['info']
            batch_data = {'inputs': [], 'out_class': [], 'info': {'file_path': [], 'pkl_idx': []}}
            batch_data_num = 0

    def _prepare_data(self, ori_data_path):
        '''
        将类型编号，正常类为class0, 其他类从 class1 ~ class47, 将数据存入文件夹名称为类名的不同文件中
        数据信息统一存入data_info.pkl
        data_info.pkl format example：
        {
            '{class_name}':{
                'l' : {id},
                'img_infos':[
                    {
                        'img_name': {string, name of image file, exclude ".jpg"},
                        'rects': [
                            [min_x, min_y, max_x, max_y],
                            ...
                        ]
                    },
                    ...
                ],
                'img_num': {integer, length of img_infos}
            }
        }
        '''
        # print(os.path.join(ori_data_path,'*train_part*'))
        ori_data_dirs = glob(os.path.join(ori_data_path, '*train_part*'))

        if len(ori_data_dirs) == 0:
            print('cannot found ori data dirs!!!!')
            assert(False)
        
        class_info_dict = {}
        class_label_idx_count = 1
        for ori_data_dir in ori_data_dirs:
            # print(ori_data_dir)
            if not os.path.isdir(ori_data_dir):
                continue
            class_names = os.listdir(ori_data_dir)
            for class_name in class_names:
                # print(class_name)
                class_path = os.path.join(ori_data_dir, class_name)

                img_paths = glob(os.path.join(class_path, '*.jpg'))

                img_infos = []

                for img_path in img_paths:
                    # print(img_path)
                    img_name = os.path.split(img_path)[-1].split('.jpg')[0]
                    if class_name == "正常":
                        img_infos.append({'img_path': img_path, 'img_name': img_name, 'rects': []})
                    else:
                        img_info_tree = ET.parse(os.path.join(class_path, img_name + '.xml'))
                        img_info_tree_root = img_info_tree.getroot()
                        rects = get_object_rect(img_info_tree_root)
                        img_infos.append({'img_path': img_path, 'img_name': img_name, 'rects': rects})

                if class_name not in class_info_dict:
                    if class_name == '正常':
                        class_label = 0
                    else:
                        class_label = class_label_idx_count
                        class_label_idx_count += 1
                    class_info_dict[class_name] = {'l': class_label, 'img_infos': img_infos, 'img_num': len(img_infos)}
                else:
                    class_info_dict[class_name]['img_infos'] += img_infos
                    class_info_dict[class_name]['img_num'] += len(img_infos)
        
        for key in class_info_dict:
            class_data_dir = os.path.join(self.data_dir, 'class%d' % class_info_dict[key]['l'])
            if not os.path.exists(class_data_dir):
                os.mkdir(class_data_dir)
            for img_info in class_info_dict[key]['img_infos']:
                print(img_info['img_path'])
                shutil.copy(img_info['img_path'], os.path.join(class_data_dir, img_info['img_name'] + '.jpg'))
                del img_info['img_path']
        
        with open(os.path.join(self.data_dir, 'data_info.pkl'), 'wb') as fp:
            pickle.dump(class_info_dict, fp)

    def _get_sample_points_per_img(self, img_size_xy, class_rects, sample_point_num = None):
        '''
        从一张图片中获取采样点
        '''
        if class_rects is None or len(class_rects) == 0:
            # 正常类，没有位置
            if sample_point_num is None:
                sample_point_num = 50
            x_sample_list = np.random.choice(list(range(img_size_xy[0])), size = sample_point_num)
            y_sample_list = np.random.choice(list(range(img_size_xy[1])), size = sample_point_num)
            
            sample_list_xy = [(y_sample_list[i], x_sample_list[i]) for i in range(sample_point_num)]

            return sample_list_xy
        else:
            # 暂时在非正常类的图片中只采样瑕疵点
            rect_point_list = [(x, y) for rect in class_rects for y in range(rect[1], rect[3]) for x in range(rect[0], rect[2])]
            area = 0
            for rect in class_rects:
                area += (rect[3] - rect[1]) * (rect[2] - rect[0])
            rect_sample_point_num = min(500, max(int(area * 0.005), 50))
            rect_sample_idx_list = np.random.choice(list(range(len(rect_point_list))), size = rect_sample_point_num, replace = False)
            rect_sample_points = []
            for idx in rect_sample_idx_list:
                rect_sample_points.append(rect_point_list[idx])

            return rect_sample_points

            # x_sample_list = np.random.choice(range(img_size_xy[0]), size = sample_point_num)
            # y_sample_list = np.random.choice(range(img_size_xy[1]), size = sample_point_num)
            
            # sample_list_yx = [(x_sample_list[i], y_sample_list[i]) for i in range(sample_point_num)]

    def _get_sample_point(self):
        '''
        获取采样点信息
        '''
        if self.sample_points_df is None:
            sample_point_csv_path = os.path.join(self.data_dir, 'sample_point.pkl')
            if not os.path.exists(sample_point_csv_path):
                print('cannot found sample point record csv, try to generate one')
                sample_point_csv = {'img_path': [], 'coordX': [], 'coordY': [], 'l': []}
                for key in self.data_info:
                    for img_info in self.data_info[key]['img_infos']:
                        img_path = os.path.join('class%d' % self.data_info[key]['l'], img_info['img_name'] + '.jpg')
                        print('img:%s' % img_path)
                        sample_point_list = self._get_sample_points_per_img((1920, 2560), img_info['rects'])
                        print('sample point num:%d' % len(sample_point_list))
                        for sample_point in sample_point_list:
                            sample_point_csv['img_path'].append(img_path)
                            sample_point_csv['coordX'].append(sample_point[0])
                            sample_point_csv['coordY'].append(sample_point[1])
                            sample_point_csv['l'].append(self.data_info[key]['l'])
                self.sample_points_df = pd.DataFrame(sample_point_csv)
                self.sample_points_df = self.sample_points_df.drop_duplicates()
                print('generate sample point info end, save as %s' % sample_point_csv_path)
                self.sample_points_df.to_csv(sample_point_csv_path, index = False)
            else:
                print('use sample point record:%s' % sample_point_csv_path)
                self.sample_points_df = pd.read_csv(sample_point_csv_path)
        
        return self.sample_points_df
        
    def split_data(self, eval_rate = 0.1, test_rate = 0.1):
        sample_point_df = self._get_sample_point()
        point_num = len(sample_point_df)
        sample_point_df = sample_point_df.sample(n = point_num)
        train_sample_point_df = sample_point_df.iloc[0 : int(point_num * ( 1 - eval_rate - test_rate))]
        eval_sample_point_df = sample_point_df.iloc[int(point_num * ( 1 - eval_rate - test_rate)) : int(point_num * ( 1 - test_rate))]
        test_sample_point_df = sample_point_df.iloc[int(point_num * ( 1 - test_rate)) : ]

        train_sample_point_df.to_csv(os.path.join(self.data_dir, 'train_sample.csv'), index = False)
        eval_sample_point_df.to_csv(os.path.join(self.data_dir, 'eval_sample.csv'), index = False)
        test_sample_point_df.to_csv(os.path.join(self.data_dir, 'test_sample.csv'), index = False)

    def _pack_data_to_pkl(self, pkl_data_dir, sample_point_df, package_num, file_name):
        if not os.path.exists(pkl_data_dir):
            os.mkdir(pkl_data_dir)
        
        # sample_point_df['img_name'] = list(map(lambda img_path: os.path.split(img_path.replace('\\', '/'))[1], sample_point_df['img_path'])) # ubuntu
        sample_point_df['img_name'] = list(map(lambda img_path: os.path.split(img_path)[1], sample_point_df['img_path'])) # windoes
        sample_point_df = sample_point_df.sort_values(by = ['img_name'])
        sample_num_per_pkl = (len(sample_point_df) + package_num - 1) // package_num
        img_path = None
        img_array = None
        pkl_file = []
        pkl_file_num = 0
        pkl_file_idx = 0
        data_info_pkl = []
        for _, row in sample_point_df.iterrows():
            if img_path is None or img_path != row['img_path']:
                img_array = cv2.imread(os.path.join(self.data_dir, row['img_path']))
                img_path = row['img_path']
                print(img_path)
            pkl_file.append({'data': get_img_cubic(img_array, [row['coordX'], row['coordY']]), 'l': row['l']})
            pkl_file_num += 1
            if pkl_file_num == sample_num_per_pkl:
                pkl_file_path = os.path.join(pkl_data_dir, file_name + ('%d' % pkl_file_idx) + '.pkl')
                with open(pkl_file_path, 'wb') as fp:
                   pickle.dump(pkl_file, fp)
                pkl_file = []
                
                pkl_file_idx += 1
                data_info_pkl.append({'file_name': os.path.split(pkl_file_path)[-1], 'data_num': pkl_file_num})
                print(pkl_file_path)
                print('data num: %d' % pkl_file_num)
                pkl_file_num = 0
        if pkl_file_num:
            pkl_file_path = os.path.join(pkl_data_dir, file_name + ('%d' % pkl_file_idx) + '.pkl')
            with open(pkl_file_path, 'wb') as fp:
               pickle.dump(pkl_file, fp)
            pkl_file = []
            pkl_file_idx += 1
            data_info_pkl.append({'file_name': os.path.split(pkl_file_path)[-1], 'data_num': pkl_file_num})
            print(pkl_file_path)
            print('data num: %d' % pkl_file_num)
            pkl_file_num = 0
        
        with open(os.path.join(pkl_data_dir, file_name + '_info.pkl'), 'wb') as fp:
            pickle.dump(data_info_pkl, fp)

    def pre_pkl_data(self, pkl_file_dir):
        if not os.path.exists(self.train_sample_csv_path):
            print('cannot found train data sample points info, try to run init step first!!!')
            assert(False)
        if not os.path.exists(self.eval_sample_csv_path):
            print('cannot found eval data sample points info, try to run init step first!!!')
            assert(False)
        if not os.path.exists(self.test_sample_csv_path):
            print('cannot found test data sample points info, try to run init step first!!!')
            assert(False)
        self.train_data_info_df = pd.read_csv(self.train_sample_csv_path)
        self.eval_data_info_df = pd.read_csv(self.eval_sample_csv_path)
        self.test_data_info_df = pd.read_csv(self.test_sample_csv_path)
        self._pack_data_to_pkl(pkl_file_dir, self.train_data_info_df, 20, 'train')
        self._pack_data_to_pkl(pkl_file_dir, self.eval_data_info_df, 2, 'eval')
        self._pack_data_to_pkl(pkl_file_dir, self.test_data_info_df, 2, 'test')

    def _get_data_num(self, pkl_data_info):
        total_num = 0
        for data_info in pkl_data_info:
            total_num += data_info['data_num']
        return total_num
    
    def get_train_data_num(self):
        return self._get_data_num(self.train_pkl_data_info)

    def get_eval_data_num(self):
        return self._get_data_num(self.eval_pkl_data_info)
    
    def get_test_data_num(self):
        return self._get_data_num(self.test_pkl_data_info)
    
    def show_some_example_img(self, pkl_path, target_dir):
        with open(pkl_path, 'rb') as fp:
            data_file = pickle.load(fp)
        idx = 0
        np.random.shuffle(data_file)
        for record in data_file:
            cv2.imwrite(os.path.join(target_dir, 'img_%d_%d.png' % (idx, record['l'])), record['data'])
            idx += 1
            if idx >= 100:
                break

if __name__ == '__main__':
    if False:
        data_set = DataSet("D:\\data\\tc", True, 'D:\\data\\天池')
        data_set.split_data()
        data_set.pre_pkl_data("D:\\data\\tc\\pkl_data\\")
    # with open('D:\\data\\tc', 'rb') as fp:
    #     data_info = pickle.load(fp)
    if False:
        # 通过合并文件夹后的图像总数，查看是否有重名图像，图像总数2022
        total_img_num = 0
        for class_path in glob('D:\\data\\tc\\' + 'class*'):
            total_img_num += len(os.listdir(class_path))
        
        print(total_img_num)
    
    if True:
        data_set = DataSet("D:\\data\\tc", False)
        data_set.show_some_example_img('D:\\data\\tc\\pkl_data\\train0.pkl', 'D:\\data\\tc\\pkl_data\\img\\')

        
