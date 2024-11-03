import os
import csv
import copy
import itertools
from collections import deque
import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# Directory containing gesture folders (A, B, C, ...)
gesture_data_dir = 'images'

# Output CSV file path
csv_path = 'gen_keys1.csv'

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    max_value = max(list(map(abs, itertools.chain.from_iterable(temp_landmark_list))))
    return [n / max_value for n in itertools.chain.from_iterable(temp_landmark_list)]

def logging_csv(label, landmark_list):
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, *landmark_list])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Iterate over each gesture folder
# for label_folder in os.listdir(gesture_data_dir):
#     label_path = os.path.join(gesture_data_dir, label_folder)

#     if label_folder.startswith('.') or not os.path.isdir(label_path):
#         continue
#     print(label_path)
#     label = label_folder.upper()  # Use the folder name as the label

#     for img_file in os.listdir(label_path):
#         # img_path = os.path.join(label_path, img_file)
#         if img_file.startswith('.') or not os.path.isdir(img_file):
#             continue
#         print(img_file)
#         image = cv.imread(img_file)
#         if image is None:
#             print(f"Could not read image {img_file}")
#             continue

#         # Process image with MediaPipe Hands
#         image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#         results = hands.process(image)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 landmark_list = calc_landmark_list(image, hand_landmarks)
#                 pre_processed_landmark_list = pre_process_landmark(landmark_list)
#                 logging_csv(label, pre_processed_landmark_list)
cnt = 0
ls=[]

marker=0
marker1=0

for label_folder in os.listdir(gesture_data_dir):
    label_path = os.path.join(gesture_data_dir, label_folder)
    # Skip hidden files or any non-directory entries
    if label_folder.startswith('.') or not os.path.isdir(label_path) or label_folder != "M":
        continue
    ls.append(label_folder)
    print(label_folder)
    
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)

        marker += 1
        print(img_path)
        image = cv.imread(img_path)
        # print(image)

        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                # print(landmark_list)
                # break

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                # Write to the dataset file
                logging_csv(cnt,pre_processed_landmark_list)
            
    cnt += 1
    break


print(ls)
print(marker1)
hands.close()
print("Dataset generation complete.")


