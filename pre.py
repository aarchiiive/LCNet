import os
import glob
from tqdm import tqdm

import cv2

num_classes = 8
path_name = "datasets/pre"

for i in range(num_classes):
    # 원본 이미지 폴더와 훈련/검증 데이터 저장 폴더 설정
    image_folder = os.path.join(path_name, str(i))  # 원본 이미지 폴더 경로
    train_folder = os.path.join(path_name, "train", str(i))  # 훈련 데이터 폴더 경로
    val_folder = os.path.join(path_name, "val", str(i))  # 검증 데이터 폴더 경로
    
    train_images = glob.glob(os.path.join(train_folder, "*.jpg"))
    val_images = glob.glob(os.path.join(val_folder, "*.jpg"))
    
    for img in tqdm(train_images):
        im = cv2.imread(img)
        im = cv2.resize(im, (512, 512))
        cv2.imwrite(img, im)
        
    for img in tqdm(val_images):
        im = cv2.imread(img)
        im = cv2.resize(im, (512, 512))
        cv2.imwrite(img, im)