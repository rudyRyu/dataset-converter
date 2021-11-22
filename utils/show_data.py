import os

import cv2
from tqdm import tqdm


def show_voc_txt(label_file, image_dir):
  with open(label_file, "r") as f:
    for line in f.readlines():
      split = line.split()
      file_path = os.path.join(image_dir, split[0])

      img = cv2.imread(file_path)

      if img is None:
        assert False

      for rect in split[1:]:
        x1, y1, x2, y2 = list(map(int, rect.split(',')))[:4]

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)

      print(file_path)
      # w = 512
      # h = 384

      # img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LANCZOS4)
      cv2.imshow('img', img)
      cv2.waitKey()

def show_yolov5(input_dir):
  image_dir = os.path.join(input_dir, 'images')
  label_dir = os.path.join(input_dir, 'labels')

  if not os.path.exists(image_dir):
    raise FileNotFoundError(f'image dir {image_dir} does not exist')
  
  if not os.path.exists(label_dir):
    raise FileNotFoundError(f'label dir {label_dir} does not exist')
  
  img_exts = ['.jpg', '.png', '.jpeg']

  p_bar = tqdm(total=len(os.listdir(image_dir)), desc='spliting')
  for image_file in os.listdir(image_dir):
    print(image_file)
    image_path = os.path.join(image_dir, image_file)
    image_name, ext = os.path.splitext(image_file)
    if ext.lower() not in img_exts:
      continue
    
    label_path = os.path.join(label_dir, image_name+'.txt')
    if not os.path.exists(label_path):
      continue
    
    image = cv2.imread(image_path)

    fx = 360 / image.shape[1]
    fy = fx
    image = cv2.resize(image, None, fx=fx, fy=fy)
    with open(label_path, 'r') as f:
      for line in f.readlines():
        label, cx, cy, w, h = map(float, line.split())
        x1 = int((cx - w/2)*image.shape[1])
        y1 = int((cy - h/2)*image.shape[0])
        x2 = int((cx + w/2)*image.shape[1])
        y2 = int((cy + h/2)*image.shape[0])

        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 3)
    
    cv2.imshow('image', image)
    cv2.waitKey()
        
    p_bar.update(1)


if __name__ == '__main__':
  # show_voc_txt(
  #   label_file='/Users/rudy/Desktop/Development/virtualenv/dataset-converter/total(dk+sh+google)_recognizable_plate.txt',
  #   image_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/total(dk+sh+google)')

  show_yolov5(
    '/Users/rudy/Desktop/Development/dataset/digits/panel_data/origin_data/yolov5'
  )