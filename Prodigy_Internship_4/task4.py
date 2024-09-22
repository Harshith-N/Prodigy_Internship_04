import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Define constants
root_dir = '/Users/harshithn/Desktop/Prodigy_Internship_4/archive/leapGestRecog'  # Update to your actual dataset path
input_shape = (100, 100)  # Resize images to 100x100 for example

gesture_folders = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
                   '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

# Custom dataset class
class HandGestureDataset(Dataset):
    def __init__(self, root_dir, input_shape):
        self.root_dir = root_dir
        self.input_shape = input_shape
        self.data, self.labels = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label

    def load_data(self):
        data = []
        labels = []

        for subject_id in range(10):
            subject_folder = f'{subject_id:02}'
            for gesture_folder in gesture_folders:
                gesture_path = os.path.join(self.root_dir, subject_folder, gesture_folder)
                if not os.path.exists(gesture_path):
                    continue
                for img_file in os.listdir(gesture_path):
                    img_path = os.path.join(gesture_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f'Failed to read image: {img_path}')
                        continue
                    img = cv2.resize(img, self.input_shape)
                    data.append(img)
                    labels.append(gesture_folder)

        # Convert lists to numpy arrays
        data = np.array(data)
        labels = np.array(labels)

        print(f'Total images loaded: {len(data)}')
        print(f'Total labels loaded: {len(labels)}')

        return data, labels

# Define CNN model architecture
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize dataset and dataloaders
dataset = HandGestureDataset(root_dir, input_shape)

# Check shapes before splitting
print(f'Data shape: {dataset.data.shape}')
print(f'Labels shape: {dataset.labels.shape}')

# Ensure there are samples to split
if len(dataset.data) == 0:
    raise ValueError("No data loaded. Please check the dataset path and image files.")

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.labels, test_size=0.2, random_state=42)

train_dataset = []
for idx in range(len(X_train)):
    label_idx = np.where(y_train[idx] == np.array(gesture_folders))[0]
    if len(label_idx) > 0:
        label_idx = label_idx[0]
        train_dataset.append((torch.tensor(X_train[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(label_idx, dtype=torch.long)))
    else:
        print(f"Skipping sample {idx} because label {y_train[idx]} does not match any gesture folder.")

test_dataset = [(torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0), 
                 torch.tensor(np.where(y_test[idx] == np.array(gesture_folders))[0][0], dtype=torch.long)) 
                for idx in range(len(X_test))]

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = CNNModel(num_classes=len(gesture_folders))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print training statistics
    epoch_loss = running_loss / len(train_dataloader)
    epoch_acc = correct / total * 100
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# Evaluation on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {(correct/total)*100:.2f}%')

import matplotlib.pyplot as plt

# Function to visualize predictions
def visualize_predictions(model, dataloader, gesture_labels):
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted_labels = [gesture_labels[p.item()] for p in predicted]

            # Display images and predicted labels
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            for i, ax in enumerate(axes.flat):
                ax.imshow(inputs[i].squeeze(), cmap='gray')
                ax.axis('off')
                ax.set_title(f'Prediction: {predicted_labels[i]}')
            plt.tight_layout()
            plt.show()
            break  # Only visualize the first batch for brevity

# Visualize predictions on test set
visualize_predictions(model, test_dataloader, gesture_folders)
