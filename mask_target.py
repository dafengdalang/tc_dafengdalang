import numpy as np
import cv2
import os
from glob import glob
import xml.etree.ElementTree as ET

def cv_imread(file_path):
    return cv2.imdecode(np.fromfile(file_path, dtype = np.uint8), -1)


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

data_dirs = ['D:\\data\\天池\\xuelang_round1_train_part1_20180628\\', 'D:\\data\\天池\\xuelang_round1_train_part2_20180705\\', 'D:\\data\\天池\\xuelang_round1_train_part3_20180709\\']

save_dir = "D:\\data\\天池\\mask_data\\"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for data_dir in data_dirs:
    class_names = os.listdir(data_dir)
    for class_name in class_names:
        if class_name == '正常':
            continue
        class_path = data_dir + class_name + "\\"
        img_paths = glob(class_path + '*.jpg')

        class_save_path = save_dir + class_name + "\\"
        if not os.path.exists(class_save_path):
            os.mkdir(class_save_path)

        for img_path in img_paths:
            img_name = img_path.split('\\')[-1].split('.jpg')[0]
            img_info_tree = ET.parse(class_path + img_name + '.xml')
            img_info_tree_root = img_info_tree.getroot()
            rects = get_object_rect(img_info_tree_root)
            img_array = cv_imread(img_path) # 因为路径中有中文，所以不能用imread
            for rect in rects:
                cv2.rectangle(img_array, (rect[0], rect[1]), (rect[2], rect[3]), (0,0,255), 3)
            img_save_path = class_save_path + img_name + '.jpg'
            cv2.imencode('.jpg', img_array)[1].tofile(img_save_path) # 因为路径中有中文，所以不能用imwrite
            print(img_save_path)
            

        
        

        