import os

import cv2


def show_voc_txt(label_file):
    with open(label_file, "r") as f:
        for line in f.readlines():
            split = line.split()
            file_path = split[0]

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

if __name__ == '__main__':
    show_voc_txt('annotation.txt')