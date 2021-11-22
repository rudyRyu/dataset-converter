import os
import json
import shutil
from collections import defaultdict
from pprint import pprint

import cv2
from tqdm import tqdm


def ltrb_to_cxcywh(ltrb):
    l,t,r,b = ltrb

    cx = (l+r) / 2.
    cy = (t+b) / 2.
    w = (r-l)
    h = (b-t)

    return [cx,cy,w,h]

def custom_json_to_yolov5(custom_json_path, base_dir, output_dir):
    with open(custom_json_path) as f:
        label_dict = json.load(f)

    total_items = list(label_dict.items())

    output_image_dir = os.path.join(output_dir, 'images/')
    output_label_dir = os.path.join(output_dir, 'labels/')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    p_bar = tqdm(total=len(total_items), desc='converting')
    for key, value in total_items:
        # save image
        image_path = os.path.join(base_dir, key)
        image_file = os.path.basename(key)
        image_name = os.path.splitext(image_file)[0]

        output_image_path = os.path.join(output_image_dir, image_file)

        image = cv2.imread(image_path)
        if image is not None:
            shutil.copy2(image_path, output_image_path)

        # save label
        class_ids = value['detection_label']['class_ids']
        box_list = value['detection_label']['box_list']

        output_label_path = os.path.join(
            output_label_dir, image_name+'.txt')

        with open(output_label_path, 'x') as f_out:
            coord = ''
            for class_id, box in zip(class_ids, box_list):
                cx,cy,w,h = ltrb_to_cxcywh(box)
                coord += f'{class_id} {cx} {cy} {w} {h}\n'
            
            if coord:
                f_out.write(coord)
        
        p_bar.update(1)


def custom_json_to_yolov5_allinone(custom_json_path, base_dir, output_dir):
    with open(custom_json_path) as f:
        label_dict = json.load(f)

    with open(custom_json_path) as f:
        label_map_dict = json.load(f)

    total_items = list(label_dict.items())

    output_image_dir = os.path.join(output_dir, 'images/')
    output_label_dir = os.path.join(output_dir, 'labels/')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for key, value in total_items:
        if 'ksf_image' in key:
            prefix = 'ksf_'
        elif 'lp_det' in key:
            prefix = 'artoa_'
        elif 'total' in key:
            prefix = 'ij_'
        else:
            prefix = ''

        # save image
        image_path = os.path.join(base_dir, key)
        image_file = os.path.basename(key)
        image_name = os.path.splitext(image_file)[0]

        output_image_path = os.path.join(output_image_dir, prefix+image_file)

        image = cv2.imread(image_path)
        if image is not None:
            shutil.copy2(image_path, output_image_path)

        # save label
        class_ids = value['detection_label']['class_ids']
        box_list = value['detection_label']['box_list']

        output_label_path = os.path.join(
            output_label_dir, prefix+image_name+'.txt')

        with open(output_label_path, 'x') as f_out:
            coord = ''
            for class_id, box in zip(class_ids, box_list):
                cx,cy,w,h = ltrb_to_cxcywh(box)
                coord += f'{class_id} {cx} {cy} {w} {h}\n'
            
            if coord:
                f_out.write(coord)


def label_map_info(label_map_json_path):
    with open(label_map_json_path) as f:
        label_map_dict = json.load(f)

    print(label_map_dict.keys())
    print(len(label_map_dict.keys()))


if __name__ == '__main__':

    custom_json_to_yolov5(
        custom_json_path='/Users/rudy/Desktop/Development/dataset/digits/panel_data/origin_data/total.json',
        base_dir='/Users/rudy/Desktop/Development/dataset/digits/panel_data/origin_data/darknet/',
        output_dir='/Users/rudy/Desktop/Development/dataset/digits/panel_data/origin_data/yolov5'
    )

    # custom_json_to_yolov5_allinone(
    #     custom_json_path='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/plate_and_char_detection/custom_json/train_0.7.json',
    #     base_dir = '/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/plate_and_char_detection/custom_json',
    #     output_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/plate_and_char_detection/yolov5/train'
    # )

    # label_map_info('/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/char_detection_and_recognition_with_type/custom_json/label_map.json')

