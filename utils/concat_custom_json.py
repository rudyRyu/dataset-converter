import glob
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
from tqdm import tqdm

def concat_custom_json(json_paths, output_json, 
                       shuffle=False):


    new_label_dict = {}    
    for json_path in json_paths:
        with open(json_path) as f:
            label_dict = json.load(f)
            new_label_dict.update(label_dict)
        
    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(new_label_dict, f_out, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    concat_custom_json(
        json_paths=[
            '/Users/rudy/Desktop/Development/dataset/license_plate/dataset/artoa/det_data/plate_detection_train.json',
            '/Users/rudy/Desktop/Development/dataset/license_plate/dataset/plate_detection_train_ksf_02462.json',
        ],
        output_json='test.json'
    )