import pandas as pd
import numpy as np
import csv

# Load the CSV file
input_file = 'sign_mnist_test.csv'  # Replace with your input CSV filename
output_file = 'landmarks.csv'  # Replace with your desired output CSV filename

# Function to compute landmarks (e.g., keypoints from pixel data)
def compute_landmarks(pixels):
    image = np.array(pixels).reshape(28, 28)  # Adjust if different size
    landmarks = []
    
    # Example: Identify keypoints (just an example calculation, adjust as needed)
    for i in range(0, 28, 4):  # Sample every 4th row/column for simplicity
        for j in range(0, 28, 4):
            block = image[i:i+4, j:j+4]
            avg_x, avg_y = np.mean(np.nonzero(block), axis=1)  # Find mean non-zero positions
            landmarks.append((i + avg_x, j + avg_y))  # Adjust for block position
    
    # Flatten landmarks list
    flattened_landmarks = [item for pair in landmarks for item in pair]
    return flattened_landmarks

# Load the data, process each row, and save results
data = pd.read_csv(input_file)
landmark_data = []

for index, row in data.iterrows():
    label = row['label']
    pixels = row[1:].tolist()  # All pixel columns
    landmarks = compute_landmarks(pixels)  # Compute landmarks for the image
    landmark_data.append([label] + landmarks)

# Write the output to a new CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Header with dynamic number of landmark points
    header = ['label'] + [f'landmark_{i}' for i in range(1, len(landmark_data[0]))]
    writer.writerow(header)
    # Write rows
    writer.writerows(landmark_data)

print("Landmarks saved to", output_file)
