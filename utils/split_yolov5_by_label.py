import os
import shutil

from tqdm import tqdm

def split_yolov5_by_label(total_dir, output_dir):

  image_dir = os.path.join(total_dir, 'images')
  label_dir = os.path.join(total_dir, 'labels')

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
    label_file = image_name+'.txt'
    if not os.path.exists(label_path):
      continue

    with open(label_path, 'r') as f:
      for line in f.readlines():
        label = line.split()[0]
        os.makedirs(os.path.join(output_dir, label, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, label, 'labels'), exist_ok=True)
        target_image_path = os.path.join(
          output_dir, label, 'images', image_file)
        target_label_path = os.path.join(
          output_dir, label, 'labels', label_file)
        
        shutil.copyfile(image_path, target_image_path)
        shutil.copyfile(label_path, target_label_path)
    p_bar.update(1)      
  
if __name__ == '__main__':
  split_yolov5_by_label(
    total_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/yolov5_plate_type_detection_20211109',
    output_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/yolov5_plate_type_detection_20211109_split_label'
  )