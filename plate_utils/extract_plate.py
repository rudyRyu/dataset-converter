import os
import json
from pprint import pprint

import cv2


def check_tags(all_tag_list, tag_filters):
    return any(item in all_tag_list for item in tag_filters)


def main(vott_json, img_dir):
    with open(vott_json) as vott_buffer:
        vott = json.loads(vott_buffer.read())

    for v in vott['assets'].values():
        file_name = v['asset']['name']

        for region in v['regions']:
            check = check_tags(
                all_tag_list=region['tags'],
                tag_filters=['recog'])
                
            if not check:
                continue
            
            height = float(region['boundingBox']['height'])
            width = float(region['boundingBox']['width'])
            left = float(region['boundingBox']['left'])
            top = float(region['boundingBox']['top'])

            right = left + width
            bottom = top + height

            img_path = os.path.join(img_dir, file_name)
            img = cv2.imread(img_path)

            cv2.imshow('img', img)
            cv2.waitKey()

if __name__ == '__main__':
    main(
        vott_json='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/total(dk+sh+google)/target/vott-json-export/license_plate-export.json',
        img_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/total(dk+sh+google)'
    )