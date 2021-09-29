import json
import os
import random

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


if __name__ == '__main__':
    # split_voc_txt(
    #     input_total='/Users/rav/Desktop/plate_train/info/all_img_list_for_train.txt',
    #     out_train='/Users/rav/Desktop/plate_train/info/split_train_vs.txt',
    #     out_val='/Users/rav/Desktop/plate_train/info/split_valid_vs.txt')

    split_custom_json(
        total_in = 'test.json',
        train_out = 'train.json',
        test_out = 'test.json'
    )

