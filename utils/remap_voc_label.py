import os
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict
from pprint import pprint

import cv2
import numpy as np
from tqdm import tqdm

def remap_voc_label(voc_image_dir, voc_label_dir, start_index=0,
                    fixed_width=None, fixed_height=180,
                    color=(0,125,255), target_color=(0,255,0)):

  xml_list = glob.glob(os.path.join(voc_label_dir, '*.xml'))
  xml_list.sort()

  # p_bar = tqdm(total=len(xml_list), desc='remapping label')
  xml_len = len(xml_list)
  xml_idx = start_index
  obj_idx = 0
  while xml_len > xml_idx:
    print(xml_idx, xml_list[xml_idx])
    tree = ET.parse(xml_list[xml_idx])
    root = tree.getroot()
    image_file = root.find('filename').text
    image = cv2.imread(os.path.join(voc_image_dir, image_file))
    h, w = image.shape[:2]

    if fixed_height is None and fixed_width is None:
      res_h = h
      res_w = w
    else:
      if fixed_height is not None:
        res_h = fixed_height
        res_w = int(w*res_h/h)
      elif fixed_width is not None:
        res_w = fixed_width
        res_h = int(h*res_w/w)
      else:
        res_w = fixed_width
        res_h = fixed_height
      
      image = cv2.resize(image, (res_w, res_h))

    objs = root.findall('object')

    draw_image = image.copy()
    for obj in objs:
      label = obj.find('name').text
      
      x1 = int(float(obj.find('bndbox/xmin').text)/w*res_w)
      y1 = int(float(obj.find('bndbox/ymin').text)/h*res_h)
      x2 = int(float(obj.find('bndbox/xmax').text)/w*res_w)
      y2 = int(float(obj.find('bndbox/ymax').text)/h*res_h)

      cv2.rectangle(draw_image, (x1,y1), (x2,y2), color, 1)
      cv2.putText(draw_image, label, (x1+3,y2-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, color, 1, cv2.LINE_AA)

    obj_len = len(objs)
    if obj_idx == -1:
      obj_idx = obj_len - 1
    while True:
      label_image = draw_image.copy()
      obj = objs[obj_idx]
      label = obj.find('name').text
      
      x1 = int(float(obj.find('bndbox/xmin').text)/w*res_w)
      y1 = int(float(obj.find('bndbox/ymin').text)/h*res_h)
      x2 = int(float(obj.find('bndbox/xmax').text)/w*res_w)
      y2 = int(float(obj.find('bndbox/ymax').text)/h*res_h)

      cv2.rectangle(label_image, (x1,y1), (x2,y2), target_color, 4)
      cv2.putText(label_image, label, (x1+3,y2-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, target_color, 2, cv2.LINE_AA)
      cv2.imshow('draw_image', label_image)
      key = cv2.waitKey() & 0xFF
      
      if key == 2: # left arrow
        if obj_idx == 0:
          xml_idx -= 1
          obj_idx = -1
          break
        else:
          obj_idx -= 1
      elif key == 3: # right arrow
        obj_idx += 1
        if obj_idx == obj_len:
          obj_idx = 0
          xml_idx += 1
          break
      else:
        try:
          new_label = chr(key)
          obj.find('name').text = new_label
          tree.write(xml_list[xml_idx])
          
        except Exception as e:
          print(e)
          continue

        obj_idx += 1
        if obj_idx == obj_len:
          obj_idx = 0
          xml_idx += 1
          break
        else:
          draw_image = image.copy()
          for obj in objs:
            label = obj.find('name').text
            
            x1 = int(float(obj.find('bndbox/xmin').text)/w*res_w)
            y1 = int(float(obj.find('bndbox/ymin').text)/h*res_h)
            x2 = int(float(obj.find('bndbox/xmax').text)/w*res_w)
            y2 = int(float(obj.find('bndbox/ymax').text)/h*res_h)

            cv2.rectangle(draw_image, (x1,y1), (x2,y2), color, 1)
            cv2.putText(draw_image, label, (x1+3,y2-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 1, cv2.LINE_AA)
    
    # p_bar.update(1)


if __name__ == '__main__':
  remap_voc_label(
    voc_image_dir='/Users/rudy/Desktop/Development/dataset/digits/7segments_extraction/',
    voc_label_dir='/Users/rudy/Desktop/Development/dataset/digits/7segments_extraction/',
    start_index=70,
    fixed_height=180
  )