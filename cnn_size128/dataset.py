import numpy as np
import cv2
import os
from glob import glob
import xml.etree.ElementTree as ET
import shutil
import pickle

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
    
    def prepare_data(self, ori_data_path):
        '''
        将类型编号，正常类为class0, 其他类从 class1 ~ class47, 将数据存入文件夹名称为类名的不同文件中
        数据信息统一存入data_info.pkl
        data_info.pkl format example：
        {
            'class{id}':{
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
                # shutil.copy(img_info['img_path'], os.path.join(class_data_dir, img_info['img_name'] + '.jpg'))
                del img_info['img_path']
        
        with open(os.path.join(self.data_dir, 'data_info.pkl'), 'wb') as fp:
            pickle.dump(class_info_dict, fp)

if __name__ == '__main__':
    if False:
        data_set = DataSet("D:\\data\\tc", True, 'D:\\data\\天池')

    # with open('D:\\data\\tc', 'rb') as fp:
    #     data_info = pickle.load(fp)
    if False:
        # 通过合并文件夹后的图像总数，查看是否有重名图像，图像总数2022
        total_img_num = 0
        for class_path in glob('D:\\data\\tc\\' + 'class*'):
            total_img_num += len(os.listdir(class_path))
        
        print(total_img_num)

        