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

class DataSet(object):
    def __init__(self, data_dir, init = False, ori_data_dir = None):
        self.data_dir = data_dir
        if init:
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)
            self.prepare_data(ori_data_dir)
            with open(os.path.join(self.data_dir, 'data_info.pkl'), 'rb') as fp:
                self.data_info = pickle.load(fp)
        else:
            with open(os.path.join(self.data_dir, 'data_info.pkl'), 'rb') as fp:
                self.data_info = pickle.load(fp)
        
        self.sample_points_df = None

    def get_data_generator(self):
        pass

    def prepare_data(self, ori_data_path):
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


    def get_sample_points_per_img(self, img_size_xy, class_rects, sample_point_num = None):
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


    def get_sample_point(self):
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
                        sample_point_list = self.get_sample_points_per_img((1920, 2560), img_info['rects'])
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
        sample_point_df = self.get_sample_point()
        point_num = len(sample_point_df)
        sample_point_df = sample_point_df.sample(n = point_num)
        train_sample_point_df = sample_point_df.iloc[0 : int(point_num * ( 1 - eval_rate - test_rate))]
        eval_sample_point_df = sample_point_df.iloc[int(point_num * ( 1 - eval_rate - test_rate)) : int(point_num * ( 1 - test_rate))]
        test_sample_point_df = sample_point_df.iloc[int(point_num * ( 1 - test_rate)) : ]

        train_sample_point_df.to_csv(os.path.join(self.data_dir, 'train_sample.csv'), index = False)
        eval_sample_point_df.to_csv(os.path.join(self.data_dir, 'eval_sample.csv'), index = False)
        test_sample_point_df.to_csv(os.path.join(self.data_dir, 'test_sample.csv'), index = False)


if __name__ == '__main__':
    if False:
        data_set = DataSet("D:\\data\\tc", True, 'D:\\data\\天池')
    if True:
        data_set = DataSet("D:\\data\\tc", False)
        data_set.split_data()

    # with open('D:\\data\\tc', 'rb') as fp:
    #     data_info = pickle.load(fp)
    if False:
        # 通过合并文件夹后的图像总数，查看是否有重名图像，图像总数2022
        total_img_num = 0
        for class_path in glob('D:\\data\\tc\\' + 'class*'):
            total_img_num += len(os.listdir(class_path))
        
        print(total_img_num)

        