import os
import csv
import copy
import itertools
from collections import deque
import cv2 as cv
import numpy as np
import mediapipe as mp
import os


import pandas as pd

# Load the CSV file
df = pd.read_csv('gen_keys.csv', header=None)

# Count occurrences of each unique value in the first column
counts = df[0].value_counts()

print(counts)

# ls =  ['.DS_Store', 'R', 'R', 'U', 'U', '9', '9', '0', '0', '7', '7', 'I', 'I', 'N', 'N', 'G', 'G', '6', '6', 'Z', 'Z', '1', '1', '8', '8', 'T', 'T', 'S', 'S', 'A', 'A', '_', '_', 'F', 'F', 'O', 'O', 'H', 'H', 'M', 'M', 'J', 'J', 'C', 'C', 'D', 'D', 'V', 'V', 'Q', 'Q', '4', '4', 'X', 'X', '3', '3', 'E', 'E', 'B', 'B', 'K', 'K', 'L', 'L', '2', '2', 'Y', 'Y', '5', '5', 'P', 'P', 'W', 'W']




ls =  [ 'R',  'U', '9', '0', '7', 'I', 'N', 'G', '6', 'Z', '1', '8', 'T', 'S', 'A', '_', 'F', 'O', 'H', 'M', 'J', 'C', 'D', 'V', 'Q', '4', 'X', '3', 'E', 'B', 'K', 'L', '2', 'Y', '5', 'P', 'W']

print(ls[19])
print(ls[11])