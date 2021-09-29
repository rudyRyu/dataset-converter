import glob
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
from tqdm import tqdm


def convert_voc_to_custom_json(voc_dir, output_json, 
                               label_file='',
                               add_parent_dir_name=True):
    """
    Not completed yet
     - need to read lable_file instead of temp_label_id

    """
    parent_dir_name = os.path.basename(voc_dir)

    xml_list = glob.glob(os.path.join(voc_dir, '*.xml'))
    xml_list.sort()

    label_dict = defaultdict(lambda: {})

    temp_label_id = 0
    p_bar = tqdm(total=len(xml_list), desc='converting')
    for xml_path in xml_list:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_name = root.find('filename').text
        image_hwc = (
            float(root.find('size/height').text),
            float(root.find('size/width').text),
            float(root.find('size/depth').text))
        
        class_ids = []
        box_list = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = [
                float(obj.find('bndbox/xmin').text) / image_hwc[1],
                float(obj.find('bndbox/ymin').text) / image_hwc[0],
                float(obj.find('bndbox/xmax').text) / image_hwc[1],
                float(obj.find('bndbox/ymax').text) / image_hwc[0],
            ]

            class_ids.append(temp_label_id)
            box_list.append(bbox)

        label = {
            'class_ids': class_ids,
            'box_list': box_list
        }
        
        if add_parent_dir_name:
            image_name = os.path.join(parent_dir_name, image_name)
            label_dict[image_name]['detection_label'] = label

        p_bar.update(1)

    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(label_dict, f_out, ensure_ascii=False, indent=4)


def convert_voc_to_custom_json_ksf(voc_dir, output_json, 
                                   label_file='',
                                   add_parent_dir_name=True,
                               ):
    """
    Not completed yet
     - need to read lable_file instead of temp_label_id

    """
    parent_dir_name = os.path.basename(voc_dir)

    xml_list = glob.glob(os.path.join(voc_dir, '*.xml'))
    xml_list.sort()

    label_dict = defaultdict(lambda: {})

    temp_label_id = 0
    p_bar = tqdm(total=len(xml_list), desc='converting')
    for xml_path in xml_list:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_file = root.find('filename').text
        image_hwc = (
            float(root.find('size/height').text),
            float(root.find('size/width').text),
            float(root.find('size/depth').text))
        
        class_ids = []
        box_list = []
        for obj in root.findall('object'):
            label = obj.find('name').text

            if label == '번호판':
                bbox = [
                    float(obj.find('bndbox/xmin').text) / image_hwc[1],
                    float(obj.find('bndbox/ymin').text) / image_hwc[0],
                    float(obj.find('bndbox/xmax').text) / image_hwc[1],
                    float(obj.find('bndbox/ymax').text) / image_hwc[0],
                ]

                class_ids.append(temp_label_id)
                box_list.append(bbox)

        if class_ids and box_list:
            label = {
                'class_ids': class_ids,
                'box_list': box_list
            }
            
            if add_parent_dir_name:
                image_file = os.path.join(parent_dir_name, image_file)
                label_dict[image_file]['detection_label'] = label

        p_bar.update(1)

    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(label_dict, f_out, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    convert_voc_to_custom_json_ksf(
        '/Users/rudy/Desktop/Development/dataset/license_plate/dataset/ksf_image_02462',
        '/Users/rudy/Desktop/Development/dataset/license_plate/dataset/plate_detection_train_ksf_02462.json',
        add_parent_dir_name=True
    )