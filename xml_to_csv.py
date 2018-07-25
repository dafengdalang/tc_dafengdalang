import pandas as pd
import xml.etree.ElementTree as ET
import os
from glob import glob
import shutil

def get_object_info(tree_root):
    '''
    从xml中获取类名和位置信息
    '''
    object_info = {'names': [], 'rects': []}
    for object_elem in tree_root.iterfind('object'):
        name = object_elem.find('name').text
        for coord_elem in object_elem.iterfind('bndbox'):
            object_info['rects'].append([int(coord_elem[0].text), int(coord_elem[1].text), int(coord_elem[2].text), int(coord_elem[3].text)])
            object_info['names'].append(name)
    return object_info

def extract_csv_from_xml(data_dirs, target_dir):
    ori_data_dirs = glob(os.path.join(data_dirs, '*train_part*'))
    info_dict = {'file_name': [], 'class_name': [], 'min_x': [], 'min_y': [], 'max_x': [], 'max_y': []}
    img_dir = os.path.join(target_dir, 'images')
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for ori_data_dir in ori_data_dirs:
        if not os.path.isdir(ori_data_dir):
            continue
        class_names = os.listdir(ori_data_dir)
        for class_name in class_names:
            class_path = os.path.join(ori_data_dir, class_name)
            img_paths = glob(os.path.join(class_path, '*.jpg'))
            img_infos = []
            for img_path in img_paths:
                img_name = os.path.split(img_path)[-1].split('.jpg')[0]
                img_info_tree = ET.parse(os.path.join(class_path, img_name + '.xml'))
                img_info_tree_root = img_info_tree.getroot()
                object_info = get_object_info(img_info_tree_root)
                info_dict['file_name'] += [img_name for _ in range(len(object_info['names']))]
                info_dict['class_name'] += object_info['names']
                for rect in object_info['rects']:
                    info_dict['min_x'].append(rect[0])
                    info_dict['min_y'].append(rect[1])
                    info_dict['max_x'].append(rect[2])
                    info_dict['max_y'].append(rect[3])
                shutil.copy(img_path, os.path.join(img_dir, img_name + '.jpg'))




# ET.parse(os.path.join(class_path, img_name + '.xml')).getroot()
# img_info_tree_root = img_info_tree.getroot()