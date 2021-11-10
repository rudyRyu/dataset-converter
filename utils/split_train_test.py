import json
import os
import random
import shutil

from tqdm import tqdm

def split_voc_txt(total_in, train_out, test_ratio=0.2):
    f_in = open(total_in, "r")
    total = f_in.readlines()
    random.shuffle(total)

    train_ratio = 1-test_ratio
    train_num = int(len(total)*train_ratio)
    train = total[:train_num]
    val = total[train_num:]

    with open(train_out, 'w') as f:
        for line in train:
            f.write(line)

    with open(test_ratio, 'w') as f:
        for line in val:
            f.write(line)

def split_custom_json(total_in, train_out, test_out, 
                      test_ratio=0.2, shuffle=True):
    with open(total_in) as f:
        label_dict = json.load(f)
    
    total_items = list(label_dict.items())
    
    if shuffle:
        random.shuffle(total_items)

    total_num = int(len(total_items))
    test_num = int(total_num * test_ratio)
    test_label_dict = dict(total_items[:test_num])
    train_label_dict = dict(total_items[test_num:])

    with open(test_out, 'w', encoding='utf-8') as f_out:
        json.dump(test_label_dict, f_out, ensure_ascii=False, indent=4)

    with open(train_out, 'w', encoding='utf-8') as f_out:
        json.dump(train_label_dict, f_out, ensure_ascii=False, indent=4)

    print('total_num: ', total_num)
    print('train_num: ', total_num - test_num)
    print('test_num: ', test_num)

def split_yolov5(total_dir, output_dir, 
                 train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1,
                 shuffle=True):

    image_dir = os.path.join(total_dir, 'images')
    label_dir = os.path.join(total_dir, 'labels')

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f'image dir {image_dir} does not exist')
    
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f'label dir {label_dir} does not exist')

    total_ratio = train_ratio + valid_ratio + test_ratio
    train_ratio /= total_ratio
    valid_ratio /= total_ratio
    test_ratio /= total_ratio
    
    img_exts = ['.jpg', '.png', '.jpeg']

    total_list = []
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        image_name, ext = os.path.splitext(image_file)
        if ext.lower() not in img_exts:
            continue
        
        label_path = os.path.join(label_dir, image_name+'.txt')
        if os.path.exists(label_path):
            total_list.append((image_path, label_path))

    if shuffle:
        random.shuffle(total_list)

    total_len = len(total_list)
    train_len = int(train_ratio * total_len)
    valid_len = int(valid_ratio * total_len)
    test_len = int(test_ratio * total_len)

    if train_len:
        os.makedirs(os.path.join(output_dir, 'train/images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train/labels'), exist_ok=True)
    if valid_len:
        os.makedirs(os.path.join(output_dir, 'valid/images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'valid/labels'), exist_ok=True)
    if test_len:
        os.makedirs(os.path.join(output_dir, 'test/images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test/labels'), exist_ok=True)
    
    p_bar = tqdm(total=len(total_list), desc='spliting')
    for i, (image_path, label_path) in enumerate(total_list):
        if i < train_len:
            target_dir = 'train'
        elif i < train_len+valid_len:
            target_dir = 'valid'
        elif i < train_len+valid_len+test_len:
            target_dir = 'test'

        image_file = os.path.basename(image_path)
        label_file = os.path.basename(label_path)

        target_image_path = os.path.join(
            output_dir, target_dir, 'images', image_file)
        target_label_path = os.path.join(
            output_dir, target_dir, 'labels', label_file)

        shutil.copyfile(image_path, target_image_path)
        shutil.copyfile(label_path, target_label_path)

        p_bar.update(1)

if __name__ == '__main__':
    # split_voc_txt(
    #     input_total='/Users/rav/Desktop/plate_train/info/all_img_list_for_train.txt',
    #     out_train='/Users/rav/Desktop/plate_train/info/split_train_vs.txt',
    #     out_val='/Users/rav/Desktop/plate_train/info/split_valid_vs.txt')

    # split_custom_json(
    #     total_in='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/char_detection_and_recognition_with_type/custom_json/test_0.3.json',
    #     train_out='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/char_detection_and_recognition_with_type/custom_json/valid_0.2.json',
    #     test_out='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/char_detection_and_recognition_with_type/custom_json/test_0.1.json',
    #     test_ratio=0.3
    # )

    split_yolov5(
        total_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/yolov5_plate_type_detection_20211109',
        output_dir='/Users/rudy/Desktop/Development/dataset/license_plate/dataset/labeled/plate_type_detection_20211109'
    )

