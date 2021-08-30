import os
import json
from collections import defaultdict

import cv2


def convert_voc_txt_to_custom_json(input_txt, output_json):
    label_dict = defaultdict(lambda: {})

    with open(input_txt, 'r') as f_in:
        for line in f_in.readlines():
            split = line.split()
            file_path = split[0]
            img = cv2.imread(file_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            class_ids = []
            box_list = []
            for rect in split[1:]:
                x1, y1, x2, y2 = list(map(float, rect.split(',')))[:4]

                x1 /= w
                y1 /= h
                x2 /= w
                y2 /= h

                class_ids.append(0)
                box_list.append([x1, y1, x2, y2])

            label = {
                'class_ids': class_ids,
                'box_list': box_list
            }

            label_dict[file_path]['detection_label'] = label

    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(label_dict, f_out, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    convert_voc_txt_to_custom_json(
        input_txt='/Users/rudy/Desktop/plate_dataset/annotation.txt', 
        output_json='/Users/rudy/Desktop/plate_dataset/annotation.json'
    )