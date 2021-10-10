import glob
import json
import os
from typing import DefaultDict
import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict
from pprint import pprint

import cv2
import numpy as np
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
    label_id = 0
    temp_label_map = {}

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

            if label not in temp_label_map:
                temp_label_map[label] = label_id
                label_id += 1


            # if label != '번호판':
                
            bbox = [
                float(obj.find('bndbox/xmin').text) / image_hwc[1],
                float(obj.find('bndbox/ymin').text) / image_hwc[0],
                float(obj.find('bndbox/xmax').text) / image_hwc[1],
                float(obj.find('bndbox/ymax').text) / image_hwc[0],
            ]

            class_ids.append(temp_label_map[label])
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


def convert_voc_to_custom_json_ksf_allinone(voc_dir, output_json, 
                                            label_map_json, label_info_json,
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
    label_id = 0
    temp_label_map = {}
    temp_label_number = defaultdict(lambda: 0)

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

            if label == '0.':
                label = '0'
            if label == '1.':
                label = '1'

            temp_label_number[label] += 1

            if label not in temp_label_map:
                temp_label_map[label] = label_id
                label_id += 1
                
            bbox = [
                float(obj.find('bndbox/xmin').text) / image_hwc[1],
                float(obj.find('bndbox/ymin').text) / image_hwc[0],
                float(obj.find('bndbox/xmax').text) / image_hwc[1],
                float(obj.find('bndbox/ymax').text) / image_hwc[0],
            ]

            class_ids.append(temp_label_map[label])
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

    with open(label_map_json, 'w', encoding='utf-8') as f_out:
        json.dump(temp_label_map, f_out, ensure_ascii=False, indent=4)

    with open(label_info_json, 'w', encoding='utf-8') as f_out:
        od = OrderedDict(sorted(temp_label_number.items()))
        json.dump(od, f_out, ensure_ascii=False, indent=4)


def convert_voc_to_custom_json_ksf_plate_and_char(voc_dir, output_json, 
                                            add_parent_dir_name=True):
    """
    Not completed yet
     - need to read lable_file instead of temp_label_id

    """
    parent_dir_name = os.path.basename(voc_dir)

    xml_list = glob.glob(os.path.join(voc_dir, '*.xml'))
    xml_list.sort()

    label_dict = defaultdict(lambda: {})

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

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
                label_id = 0
            else:
                label_id = 1
                
            bbox = [
                float(obj.find('bndbox/xmin').text) / image_hwc[1],
                float(obj.find('bndbox/ymin').text) / image_hwc[0],
                float(obj.find('bndbox/xmax').text) / image_hwc[1],
                float(obj.find('bndbox/ymax').text) / image_hwc[0],
            ]

            class_ids.append(label_id)
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


def convert_voc_to_custom_json_ksf_char(voc_dir, output_dir, output_json, 
                                        output_image_dir_name):
    """
    Not completed yet
     - need to read lable_file instead of temp_label_id

    """
    output_image_dir = os.path.join(output_dir, output_image_dir_name)
    os.makedirs(output_image_dir, exist_ok=True)
    parent_dir_name = os.path.basename(voc_dir)

    xml_list = glob.glob(os.path.join(voc_dir, '*.xml'))
    xml_list.sort()

    label_dict = defaultdict(lambda: {})

    target_plate_size_wh = (360, 180)
    target_plate_ratio = target_plate_size_wh[0] / target_plate_size_wh[1]
    p_bar = tqdm(total=len(xml_list), desc='converting')
    for xml_path in xml_list:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_file = root.find('filename').text
        image_name = os.path.splitext(image_file)[0]
        image_hwc = (
            float(root.find('size/height').text),
            float(root.find('size/width').text),
            float(root.find('size/depth').text))
        image = cv2.imread(os.path.join(voc_dir, image_file))

        objs = root.findall('object')
        
        
        # Get plate list first
        plates = []
        idx = 0
        for obj in objs:
            label = obj.find('name').text
            if label == '번호판':
                plate_dict = {}
                x1 = int(obj.find('bndbox/xmin').text)
                y1 = int(obj.find('bndbox/ymin').text)
                x2 = int(obj.find('bndbox/xmax').text)
                y2 = int(obj.find('bndbox/ymax').text)
                
                plate_center_h = round((y1+y2)/2)
                plate_center_w = round((x1+x2)/2)

                plate_x1 = x1 - round((x2-x1)*0.1)
                plate_x2 = x2 + round((x2-x1)*0.1)
                plate_w = plate_x2 - plate_x1
                plate_y1 = plate_center_h - round(plate_w/4)
                plate_y2 = plate_center_h + round(plate_w/4)

                pad_left, pad_right, pad_top, pad_bottom = 0,0,0,0
                ratio = (x2-x1) / (y2-y1)
                # 세로가 큰 경우
                if ratio < target_plate_ratio:
                    plate_y1 = y1 - round((y2-y1)*0.05)
                    plate_y2 = y2 + round((y2-y1)*0.05)
                    plate_h = plate_y2 - plate_y1

                    plate_x1 = plate_center_w - plate_h
                    plate_x2 = plate_center_w + plate_h
                else:
                    plate_x1 = x1 - round((x2-x1)*0.05)
                    plate_x2 = x2 + round((x2-x1)*0.05)
                    plate_w = plate_x2 - plate_x1
                    plate_y1 = plate_center_h - round(plate_w/4)
                    plate_y2 = plate_center_h + round(plate_w/4)

                if plate_x1 < 0:
                    pad_left = -plate_x1
                    plate_x1 = 0
                if plate_x2 >= image_hwc[1]:
                    pad_right = plate_x2-image_hwc[1]
                    plate_x2 = image_hwc[1]
                if plate_y1 < 0:
                    pad_top = -plate_y1
                    plate_y1 = 0
                if plate_y2 >= image_hwc[0]:
                    pad_bottom = plate_y2-image_hwc[0]
                    plate_y2 = image_hwc[0]

                plate_image = image[plate_y1:plate_y2,
                                    plate_x1:plate_x2]


                if pad_left or pad_right or pad_top or pad_bottom:
                    plate_image = cv2.copyMakeBorder(
                        plate_image,
                        top=pad_top,
                        bottom=pad_bottom,
                        left=pad_left,
                        right=pad_right,
                        borderType=cv2.BORDER_CONSTANT,
                        value=(0,0,0)
                    )

                plate_dict['plate_image'] = plate_image
                plate_dict['plate_box'] = [x1,y1,x2,y2]
                plate_dict['new_plate_box'] = [plate_x1, plate_y1, plate_x2, plate_y2]
                plate_dict['pad'] = [pad_left, pad_top, pad_right, pad_bottom]
                plate_dict['save_image_file'] = f'{image_name}_{idx}.jpg'
                idx += 1

                plates.append(plate_dict)
                
                save_image_path = os.path.join(output_image_dir, 
                    plate_dict['save_image_file'])
                success = cv2.imwrite(save_image_path, plate_image)

                if not success:
                    print(save_image_path)
                    assert False

        # Get char in the plate
        for plate_dict in plates:
            new_plate_box = plate_dict['new_plate_box']
            plate_size_hw = plate_dict['plate_image'].shape[:2]
            pad = plate_dict['pad']
            plate_image_file = plate_dict['save_image_file']

            temp_label_id = 0
            class_ids = []
            box_list = []
            for obj in objs:
                label = obj.find('name').text
                if label == '번호판':
                    continue

                x1 = int(obj.find('bndbox/xmin').text)
                y1 = int(obj.find('bndbox/ymin').text)
                x2 = int(obj.find('bndbox/xmax').text)
                y2 = int(obj.find('bndbox/ymax').text)

                if (x1 >= new_plate_box[0] and x2 <= new_plate_box[2] 
                        and y1 >= new_plate_box[1] and y2 <= new_plate_box[3]):

                    new_x1 = int(x1 - new_plate_box[0] + pad[0])
                    new_y1 = int(y1 - new_plate_box[1] + pad[1])
                    new_x2 = new_x1 + (x2-x1)
                    new_y2 = new_y1 + (y2-y1)

                    char_box = [
                        new_x1 / plate_size_hw[1],
                        new_y1 / plate_size_hw[0],
                        new_x2 / plate_size_hw[1],
                        new_y2 / plate_size_hw[0]
                    ]

                    class_ids.append(temp_label_id)
                    box_list.append(char_box)

                    # # check padded image
                    # if pad[0] or pad[1] or pad[2] or pad[3]:
                    #     print(plate_image.shape)
                    #     print(new_x1,new_y1,new_x2,new_y2)
                    #     print(pad)
                    #     cv2.imshow('plate', plate_image)
                    #     cv2.waitKey()

            
            if class_ids and box_list:
                label = {
                    'class_ids': class_ids,
                    'box_list': box_list
                }
                
                if output_image_dir_name:
                    plate_image_file= os.path.join(output_image_dir_name, plate_image_file)
                
                label_dict[plate_image_file]['detection_label'] = label

            # # check result
            # pad = plate_dict['pad']
            # if pad[0] or pad[1] or pad[2] or pad[3]:
            #     plate_image = plate_dict['plate_image']

            #     for box in box_list:
            #         x1 = int(box[0] * plate_size_hw[1])
            #         y1 = int(box[1] * plate_size_hw[0])
            #         x2 = int(box[2] * plate_size_hw[1])
            #         y2 = int(box[3] * plate_size_hw[0])

            #         cv2.rectangle(plate_image, (x1, y1), (x2,y2), (0,255,0), 1)

            #     print(plate_dict['save_image_file'])
            #     cv2.imshow('plate_result', cv2.resize(plate_image, (720, 360)))
            #     cv2.waitKey()

        p_bar.update(1)

    # with open(output_json, 'w', encoding='utf-8') as f_out:
        # json.dump(label_dict, f_out, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    # convert_voc_to_custom_json_ksf_allinone(
    #     voc_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/ksf_image_02462',
    #     output_json='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/plate_recognition/custom_json/total.json',
    #     label_map_json='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/plate_recognition/custom_json/label_map.json',
    #     label_info_json='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/plate_recognition/custom_json/label_info.json',
    #     add_parent_dir_name=True
    # )
    
    convert_voc_to_custom_json_ksf_plate_and_char(
        voc_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/ksf_image_02462',
        output_json='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/plate_and_char_detection/custom_json/total.json',
        add_parent_dir_name=True
    )

    # convert_voc_to_custom_json_ksf_char(
    #     voc_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/ksf_image_02462',
    #     output_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/char_detection/custom_json/',
    #     output_image_dir_name='ksf_image_02462',
    #     output_json='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/char_detection/custom_json/total.json'
    # )
