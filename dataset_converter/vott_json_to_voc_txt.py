import os
import json
from pprint import pprint

import cv2


def convert_vott_to_voc(vott_json, output):
    f_out = open(output, 'w+')

    with open(vott_json) as vott_buffer:
        vott = json.loads(vott_buffer.read())

    for v in vott['assets'].values():
        file_name = v['asset']['name']

        coord = ''
        for region in v['regions']:
            # check = check_tags(region['tags'], 
            #   ['medium', 'small', 'very small'])
            # if check:
            if True:
                height = float(region['boundingBox']['height'])
                width = float(region['boundingBox']['width'])
                left = float(region['boundingBox']['left'])
                top = float(region['boundingBox']['top'])

                right = left + width
                bottom = top + height

                coord += f' {int(left)},{int(top)},{int(right)},{int(bottom)},0'

        if coord:
            f_out.write(file_name + coord + '\n')

    f_out.close()


def check_tags(all_tag_list, tag_filters):
    return any(item in all_tag_list for item in tag_filters)


def show_rect(vott_json, img_dir):
    with open(vott_json) as vott_buffer:
        vott = json.loads(vott_buffer.read())

    for i, v in enumerate(list(vott['assets'].values())[::-1]):
        file_name = v['asset']['name']

        img = cv2.imread(os.path.join(img_dir, file_name))
        for region in v['regions']:
            height = float(region['boundingBox']['height'])
            width = float(region['boundingBox']['width'])
            left = float(region['boundingBox']['left'])
            top = float(region['boundingBox']['top'])

            right = left + width
            bottom = top + height

            l, t, r, b = int(left), int(top), int(right), int(bottom)
            cv2.rectangle(img, (l,t), (r,b), (0,255,0), 1)

            # text = ', '.join(region['tags'])
            # img = cv2.putText(img, text, (l,t), cv2.FONT_HERSHEY_SIMPLEX,
            #        1, (0, 255, 0), 1, cv2.LINE_AA)


        print(i, file_name)
        cv2.imshow('img', img)
        cv2.waitKey()

if __name__ == '__main__':
    convert_vott_to_voc(
        vott_json='/Users/rav/Desktop/plate_train/dataset/ETRI/target/vott-json-export/license_plate_etri-export.json',
        output='/Users/rav/Desktop/etri_voc_output.txt')

    # show_rect(
    #     vott_json='/Users/rav/Desktop/plate_train/dataset/ETRI/target/vott-json-export/license_plate_etri-export.json',
    #     img_dir='/Users/rav/Desktop/plate_train/dataset/ETRI')