import os
import json
from collections import defaultdict


def darknet_to_custom_json(input_txt, data_dir, output_json):
    label_dict = defaultdict(lambda: {})

    f = open(input_txt, 'r')
    
    for image_name in f:
        base_image_name = os.path.basename(image_name.strip())
        label_filename = os.path.join(data_dir, 
            os.path.splitext(base_image_name)[0] + '.txt')

        f_label = open(label_filename, 'r')
        
        class_ids = []
        boxes = []
        for label in f_label:
            class_id, cx, cy, w, h = label.split(' ')

            cx, cy, w, h = tuple(map(float, (cx, cy, w, h)))

            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            
            class_ids.append(int(class_id))
            boxes.append([x1, y1, x2, y2])

        label = {
            'class_ids': class_ids,
            'box_list': boxes
        }

        label_dict[base_image_name]['detection_label'] = label

    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(label_dict, f_out, ensure_ascii=False, indent=4)

    f.close()

if __name__ == '__main__':
    darknet_to_custom_json(
        input_txt='/Users/rudy/Desktop/Development/dataset/digits/panel_data/origin_data/total.txt',
        data_dir='/Users/rudy/Desktop/Development/dataset/digits/panel_data/origin_data/data',
        output_json='/Users/rudy/Desktop/Development/dataset/digits/panel_data/origin_data/total.json'
    )