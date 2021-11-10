import json
import os
import random
import shutil
import xml.etree.ElementTree as ET



def copy_valid_voc_data(input_dir, output_dir):
  """
  copy images containing label file that has valid bounding box 
  """

  if not os.path.exists(input_dir):
    raise FileNotFoundError(f'input_dir {input_dir} does not exist')

  os.makedirs(output_dir, exist_ok=True)

  img_exts = ['.jpg', '.png', '.jpeg']
  for image_file in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_file)
    image_name, ext = os.path.splitext(image_file)
    if ext.lower() not in img_exts:
      continue
    
    label_path = os.path.join(input_dir, image_name+'.xml')
    if not os.path.exists(label_path):
      continue

    label_file = os.path.basename(label_path)
    target_image_path = os.path.join(output_dir, image_file)
    target_label_path = os.path.join(output_dir, label_file)
      
    shutil.copyfile(image_path, target_image_path)
    shutil.copyfile(label_path, target_label_path)

if __name__ == '__main__':
  copy_valid_voc_data(
    input_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/no_labeled/netvision_gray_43000',
    output_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/no_labeled/netvision_gray_verified'
  )
