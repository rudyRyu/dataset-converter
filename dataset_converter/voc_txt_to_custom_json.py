import os
import json
from collections import defaultdict

import cv2


def convert_voc_txt_to_custom_json(input_txt, image_dir, output_json,
                                   label_file='',
                                   add_parent_dir_name=True):
    label_dict = defaultdict(lambda: {})

    parent_dir_name = os.path.basename(image_dir)

    with open(input_txt, 'r') as f_in:
        for line in f_in.readlines():
            split = line.split()

            image = cv2.imread(os.path.join(image_dir, split[0]))
            if image is None:
                continue

            h, w = image.shape[:2]
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

            if add_parent_dir_name:
                image_name = os.path.join(parent_dir_name, split[0])
            else:
                image_name = split[0]

            label_dict[image_name]['detection_label'] = label

    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(label_dict, f_out, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    convert_voc_txt_to_custom_json(
        input_txt='/Users/rudy/Desktop/Development/virtualenv/dataset-converter/total(dk+sh+google)_recognizable_plate.txt',
        image_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/total(dk+sh+google)', 
        output_json='/Users/rudy/Desktop/Development/virtualenv/dataset-converter/total(dk+sh+google)_recognizable_plate.json',
        add_parent_dir_name=True
    )