import os
import json
from pprint import pprint

import cv2
from tqdm import tqdm


def check_tags(all_tag_list, tag_filters):
  return any(item in all_tag_list for item in tag_filters)


def extract_plate(image, box, target_plate_ratio):
  image_hwc = image.shape
  x1 = int(box[0])
  y1 = int(box[1])
  x2 = int(box[2])
  y2 = int(box[3])
  
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

  return plate_image

def vott(vott_json, img_dir):
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


def yolov5(input_dir, output_dir):
  image_dir = os.path.join(input_dir, 'images')
  label_dir = os.path.join(input_dir, 'labels')

  if not os.path.exists(image_dir):
    raise FileNotFoundError(f'image dir {image_dir} does not exist')
  
  if not os.path.exists(label_dir):
    raise FileNotFoundError(f'label dir {label_dir} does not exist')

  img_exts = ['.jpg', '.png', '.jpeg']

  p_bar = tqdm(total=len(os.listdir(image_dir)), desc='spliting')
  for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    image_name, ext = os.path.splitext(image_file)
    if ext.lower() not in img_exts:
      continue
  
    label_path = os.path.join(label_dir, image_name+'.txt')
    if not os.path.exists(label_path):
      continue

    image = cv2.imread(image_path)
    with open(label_path, 'r') as f:
      for i, line in enumerate(f.readlines()):
        label, cx_f, cy_f, w_f, h_f = line.split()
        cx_f = float(cx_f)
        cy_f = float(cy_f)
        w_f = float(w_f)
        h_f = float(h_f)

        x1 = int((cx_f-w_f/2)*image.shape[1])
        y1 = int((cy_f-h_f/2)*image.shape[0])
        x2 = int((cx_f+w_f/2)*image.shape[1])
        y2 = int((cy_f+h_f/2)*image.shape[0])
        
        plate_image = extract_plate(image.copy(), [x1,y1,x2,y2], 2.0)

        new_image_dir = os.path.join(output_dir, label)
        os.makedirs(new_image_dir, exist_ok=True)
        new_image_file = f'{image_name}_{i}.jpg'
        new_image_path = os.path.join(new_image_dir, new_image_file)

        cv2.imwrite(new_image_path, plate_image)
    
    p_bar.update(1)


if __name__ == '__main__':
  # vott(
  #   vott_json='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/total(dk+sh+google)/target/vott-json-export/license_plate-export.json',
  #   img_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/total(dk+sh+google)'
  # )

  yolov5(
    input_dir='/Users/rudy/Desktop/data01',
    output_dir='/Users/rudy/Desktop/data01_extract'
  )