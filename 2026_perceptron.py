import PIL
import numpy as np
import os
import random 
from PIL import Image
from numpy import asarray

#--------------------------------------Functions--------------------------
#FORWARD PROP
#takes the summation of w1 and a single image's data
def forward_prop(weight, data, column):
    summation = np.dot(weight, data[:, column]) #take summation
    pred = np.tanh(summation) #output decimal between -1 and 1

    if pred >= 0:
        return 1 
    else:
        return -1 

#BACK PROP
alpha = 0.002 #learning rate

def back_prop(weight, classification, data_num, column):
    weight = weight + alpha * (classification * data_num[:, column]) #new w1 is w1 (matrix) + truth (scalar) * 784 (matrix)
    #'L' values of image _
    return weight

#TRAINING DATA
def label_data(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('.') and f.endswith('.jpg')] #filter out hidden files and non-jpg files
    files.sort() #sorts list of files in dir.

    # 2-step renaming to avoid overwriting/deleting files
    temp_files = []
    for filename in files:
        old_filepath = os.path.join(folder_path, filename)

        temp_filename = f"temp_{filename}"
        temp_filepath = os.path.join(folder_path, temp_filename)

        os.rename(old_filepath, temp_filepath)
        temp_files.append(temp_filename)
                
    for index, temp_filename in enumerate(temp_files, start = 1): 
        temp_filepath = os.path.join(folder_path, temp_filename) #tell Python where file is, not just name
        new_filename = f"1_{index:04d}.jpg" #pad index with leading zeros to 4 digits *change to 0,1,2,3... based on data num
        new_filepath = os.path.join(folder_path, new_filename)
        os.rename(temp_filepath, new_filepath) 

    files.sort()

# clean up data
    # label_data('/Users/xiao/Desktop/AI Implementation Consulting/Neural Network/0')
    # label_data('/Users/xiao/Desktop/AI Implementation Consulting/Neural Network/1')

def create_training_data(folder_0, folder_1):
    dataset = [] # Empty list
    def process_folder(folder_path, label):
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                filepath = os.path.join(folder_path, filename)
                image = Image.open(filepath)
                data = np.asarray(image) # Convert image into NumPy array
                var = data.reshape(-1, 1) # Flatten to 784 x 1 (MNIST dataset is 28x28 pixels)
                dataset.append((var, label)) 
    # Process both folders
    process_folder(folder_0, label = 1)
    process_folder(folder_1, label = -1)

    # Scramble dataset
    random.shuffle(dataset)

    x_list = [item[0] for item in dataset]
    y_list = [item[1] for item in dataset]

    # Convert to final NumPy matrices
    i_data = np.hstack(x_list) # Stack horizontally to get 784 x N
    ground_truth = np.array(y_list) 

    return i_data, ground_truth 

save_path = '/Users/xiao/Desktop/AI Implementation Consulting/Neural Network/mnist_data.npz'
if os.path.exists(save_path):
    print("Loading compiled data from file...")
    saved_data = np.load(save_path)

    i_data = saved_data['x']
    ground_truth = saved_data['y']

else: 
    print("Compiling data from JPGs for the first time...")
    i_data, ground_truth = create_training_data(
        '/Users/xiao/Desktop/AI Implementation Consulting/Neural Network/0',
        '/Users/xiao/Desktop/AI Implementation Consulting/Neural Network/1'
    )
    np.savez_compressed(save_path, x=i_data, y=ground_truth)
    print("Data saved successfully!")

print(f"Image matrix shape: {i_data.shape}")
print(f"Labels array shape: {ground_truth.shape}")

#-------------------------------------Saving Weights-----------------------------

weights_path = '/Users/xiao/Desktop/AI Implementation Consulting/Neural Network/trained_weights.npy'

# Check if the model has already been trained
if os.path.exists(weights_path):
    print("\nLoading pre-trained weights...")
    weights = np.load(weights_path)

# If no weights exist, train model from scratch
else:
    print("\nNo saved weights found. Starting Training (6000 Images)...")
    #-------------------------------------Parameters-----------------------------
    np.random.seed(7) #ensures weight is the same randomized matrix 
    weights = np.random.randn(1, 784) # 1x784 matrix of random weights 

    # Split data
    total_images = i_data.shape[1]
    train_size = 6000
    epochs = 5

    print("\n-- Starting Training (6000 Images) ---")

    for epoch in range(epochs): 
        correct_pred = 0

        for i in range(train_size):
            truth = ground_truth[i]
            prediction = forward_prop(weights, i_data, i)

            if prediction == truth:
                correct_pred += 1
            else:
                weights = back_prop(weights, truth, i_data, i)

        train_accuracy = (correct_pred / train_size) * 100
        print(f"Epoch {epoch + 1} | Training Accuracy: {train_accuracy: .2f}%")

    # Save trained weights to file
    np.save(weights_path, weights)
    print("Training complete. Weights saved successfully.")

print(weights)

#-------------------------------------Testing-----------------------------
'''test_correct = 0
test_total = total_images - train_size

for i in range(train_size, total_images):
    truth = ground_truth[i]
    prediction = forward_prop(weights, i_data, i)

    if prediction == truth:
        test_correct += 1

test_accuracy = (test_correct / test_total) * 100
print(f"Testing Accuracy: {test_accuracy: .2f}%")

# Results:

Loading compiled data from file...
Image matrix shape: (784, 8816)
Labels array shape: (8816,)

-- Starting Training (6000 Images) ---
Epoch 1 | Training Accuracy:  99.50%
Epoch 2 | Training Accuracy:  99.85%
Epoch 3 | Training Accuracy:  99.87%
Epoch 4 | Training Accuracy:  99.87%
Epoch 5 | Training Accuracy:  99.93%
Epoch 6 | Training Accuracy:  99.93%
Epoch 7 | Training Accuracy:  100.00%
Epoch 8 | Training Accuracy:  100.00%
Epoch 9 | Training Accuracy:  100.00%
Epoch 10 | Training Accuracy:  100.00%
Training complete. Weights saved successfully.

--- Starting Testing (2816 Unseen Images for previously 5 epochs) ---
Testing Accuracy:  99.64%

'''

# MNIST dataset: https://www.kaggle.com/datasets/scolianni/mnistasjpg?resource=download