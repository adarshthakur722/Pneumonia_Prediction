# Pneumonia Detection from Chest X-rays using CNN (PyTorch)

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify chest X-ray images as **Pneumonia** or **Normal**. The model achieves an accuracy of **90%** on the test set.

## 🔍 Project Overview

Pneumonia is a serious lung infection that requires timely diagnosis. Chest X-rays are a commonly used method for detecting pneumonia. This deep learning model automates the classification process, assisting healthcare professionals in faster diagnosis.

## 🧠 Model Architecture

The CNN model consists of 3 convolutional blocks for feature extraction followed by fully connected layers for binary classification.

### Model Summary:

```python
class NN(nn.Module):
    def __init__(self, input_feature):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_feature, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32768, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
