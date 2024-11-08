---
title: "20 Torchvision concepts with Before-and-After Examples"
seoTitle: "20 Torchvision concepts with Before-and-After Examples"
seoDescription: "20 Torchvision concepts with Before-and-After Examples"
datePublished: Fri Oct 04 2024 11:23:21 GMT+0000 (Coordinated Universal Time)
cuid: cm1umzu9r001s09l85reph71m
slug: 20-torchvision-concepts-with-before-and-after-examples
tags: image-processing, computer-vision, deep-learning, pytorch, torchvision

---

### 1\. **Installing torchvision via pip 📦**

**Boilerplate Code**:

```bash
pip install torchvision
```

**Use Case**: Install the `torchvision` library to work with computer vision models and datasets.

**Goal**: Set up `torchvision` for image processing and pre-trained model usage. 🎯

**Sample Code**:

```bash
pip install torchvision
```

**Before Example**: You manually handle image data using custom functions or external libraries.

```bash
# Handling images using external libraries like Pillow:
from PIL import Image
img = Image.open("image.jpg")
```

**After Example**: With `torchvision` installed, you can leverage pre-built datasets and image transforms.

```bash
Successfully installed torchvision
# Torchvision is ready for image transformations and model usage.
```

---

### 2\. **Loading a Pre-trained Model with torchvision 🧠**

**Boilerplate Code**:

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()
```

**Use Case**: Load a pre-trained model for image classification.

**Goal**: Use a pre-trained ResNet model to perform image classification tasks. 🎯

**Sample Code**:

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()
```

**Before Example**: You manually build and train models from scratch, which is time-consuming.

```bash
# Building a CNN from scratch:
import torch.nn as nn
model = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU(), nn.MaxPool2d(2))
```

**After Example**: With `torchvision`, you can directly use a pre-trained ResNet model for classification.

```bash
ResNet-18 model loaded with pre-trained weights.
# Ready to classify images using a state-of-the-art model.
```

---

### 3\. **Transforming Images for Model Input 🖼️**

**Boilerplate Code**:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Use Case**: Preprocess images before feeding them into a neural network.

**Goal**: Apply necessary transformations (resize, crop, normalize) for input to the model. 🎯

**Sample Code**:

```python
from torchvision import transforms
from PIL import Image

image = Image.open("image.jpg")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image)
print(image_tensor.shape)
```

**Before Example**: You manually apply each transformation to images using custom code or external libraries.

```bash
# Resizing and normalizing an image manually:
image = Image.open("image.jpg").resize((256, 256))
```

**After Example**: `torchvision` provides a flexible transformation pipeline that makes it easy to preprocess images.

```bash
torch.Size([3, 224, 224])
# Image transformed and ready for model input.
```

---

### 4\. **Loading Image Datasets with torchvision 🗂️**

**Boilerplate Code**:

```python
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

dataset = ImageFolder(root="data/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**Use Case**: Load image datasets using `ImageFolder` and apply transformations.

**Goal**: Efficiently load and batch image datasets for training or inference. 🎯

**Sample Code**:

```python
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageFolder(root="data/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    print(images.shape, labels.shape)
```

**Before Example**: You write custom code to load, batch, and transform images manually.

```bash
# Custom code for loading and batching images:
import os
image_files = os.listdir("data/train")
```

**After Example**: With `torchvision`, you can quickly load and batch datasets using a standardized interface.

```bash
torch.Size([32, 3, 224, 224]) torch.Size([32])
# Dataset loaded and batched using `torchvision`.
```

---

### 5\. **Visualizing Image Data with torchvision 📊**

**Boilerplate Code**:

```python
import matplotlib.pyplot as plt
import torchvision.utils as vutils

grid = vutils.make_grid(images, nrow=8, padding=2)
plt.imshow(grid.permute(1, 2, 0))
plt.show()
```

**Use Case**: Visualize a batch of images using `torchvision`.

**Goal**: Display multiple images in a grid format for quick visualization. 🎯

**Sample Code**:

```python
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Assuming images is a batch of tensors
grid = vutils.make_grid(images, nrow=8, padding=2)
plt.imshow(grid.permute(1, 2, 0))
plt.show()
```

**Before Example**: You manually create visualization grids using custom code, which can be time-consuming.

```bash
# Manually creating grids using a custom function:
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i].permute(1, 2, 0))
plt.show()
```

**After Example**: With `torchvision`, you can visualize a batch of images in a grid with a single function call.

```bash
# A grid of images is displayed for easy visualization.
```

### 6\. **Data Augmentation with torchvision 📈**

**Boilerplate Code**:

```python
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5)
])
```

**Use Case**: Augment your dataset by applying random transformations during training.

**Goal**: Enhance model performance by creating diverse training data through augmentation. 🎯

**Sample Code**:

```python
from torchvision import transforms
from PIL import Image

image = Image.open("image.jpg")
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5)
])

augmented_image = augmentation(image)
augmented_image.show()
```

**Before Example**: You manually apply transformations to increase the diversity of your training data.

```bash
# Manually flipping and rotating an image:
image = image.transpose(Image.FLIP_LEFT_RIGHT)
```

**After Example**: With `torchvision`, data augmentation is seamlessly integrated into the training pipeline.

```bash
# The image is randomly flipped, rotated, and color adjusted.
# Augmented image displayed.
```

---

### 7\. **Transfer Learning with Pre-trained Models in torchvision 🔄**

**Boilerplate Code**:

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Linear(512, num_classes)
```

**Use Case**: Fine-tune a pre-trained model on a new dataset using transfer learning.

**Goal**: Use a pre-trained model and adapt it for a new classification task. 🎯

**Sample Code**:

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze the model layers

model.fc = torch.nn.Linear(512, num_classes)  # Replace the final layer for new task
print(model)
```

**Before Example**: You train deep neural networks from scratch, requiring large datasets and significant computational resources.

```bash
# Training a CNN from scratch:
model = CustomCNN()
```

**After Example**: With transfer learning, you can leverage pre-trained models for faster and more efficient training on your dataset.

```bash
ResNet-18 model loaded and modified for transfer learning with a new final layer.
# Ready to fine-tune the model on new data.
```

---

### 8\. **Saving and Loading Models with torchvision 🛠️**

**Boilerplate Code**:

```python
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
```

**Use Case**: Save and load trained models for reuse or further training.

**Goal**: Persist model weights to disk and reload them as needed. 🎯

**Sample Code**:

```python
import torch

# Save the model
torch.save(model.state_dict(), "model.pth")

# Load the model
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Set the model to evaluation mode
```

**Before Example**: You manually handle model persistence using external libraries or frameworks.

```bash
# Saving model weights manually using custom code:
pickle.dump(model_weights, open("model_weights.pkl", "wb"))
```

**After Example**: With `torchvision`, saving and loading models is streamlined and handled through PyTorch’s built-in methods.

```bash
Model weights saved to model.pth.
Model weights loaded successfully and set to evaluation mode.
```

---

### 9\. **Using torchvision’s Built-in Datasets 🗄️**

**Boilerplate Code**:

```python
from torchvision import datasets
from torch.utils.data import DataLoader

train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

**Use Case**: Load popular image datasets like CIFAR-10, ImageNet, or MNIST using `torchvision`.

**Goal**: Quickly access and load commonly used datasets for training and evaluation. 🎯

**Sample Code**:

```python
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for images, labels in train_loader:
    print(images.shape, labels.shape)
```

**Before Example**: You manually download and preprocess datasets from external sources.

```bash
# Manually downloading and loading CIFAR-10 dataset:
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

**After Example**: With `torchvision`, built-in datasets are easily accessible with automated downloading and preprocessing.

```bash
torch.Size([64, 3, 32, 32]) torch.Size([64])
# CIFAR-10 dataset downloaded, loaded, and ready for training.
```

---

### 10\. **Extracting Features with torchvision Models 🔍**

**Boilerplate Code**:

```python
model = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
```

**Use Case**: Use a pre-trained model as a feature extractor for image embeddings.

**Goal**: Extract feature representations from images using pre-trained models. 🎯

**Sample Code**:

```python
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final layer

# Preprocess an image
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
image = Image.open("image.jpg")
image_tensor = transform(image).unsqueeze(0)

# Extract features
with torch.no_grad():
    features = feature_extractor(image_tensor)
print(features.shape)
```

**Before Example**: You manually build custom feature extraction pipelines or manually extract features for individual images.

```bash
# Custom feature extraction pipeline:
image_features = extract_features(image)
```

**After Example**: With `torchvision`, you can extract high-level features from pre-trained models with minimal setup.

```bash
torch.Size([1, 512, 1, 1])
# High-level features extracted from the image using ResNet-18.
```

### 11\. **Fine-Tuning a Pre-trained Model with torchvision 🔧**

**Boilerplate Code**:

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze base layers

model.fc = torch.nn.Linear(512, num_classes)  # Modify final layer for new task
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

**Use Case**: Adapt a pre-trained model to a specific task by fine-tuning the last layers.

**Goal**: Fine-tune a pre-trained model on a new dataset while freezing most of the model's layers. 🎯

**Sample Code**:

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers except the final one

model.fc = torch.nn.Linear(512, num_classes)  # Update the final layer for the new task
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Now the model can be fine-tuned on your new dataset
```

**Before Example**: You train deep learning models from scratch, which requires a lot of data and time.

```bash
# Training a CNN from scratch:
model = CustomCNN()
```

**After Example**: With `torchvision`, you can fine-tune a pre-trained model on your dataset with fewer resources.

```bash
ResNet-18 fine-tuned with a new final layer for specific classification tasks.
# The model is now ready to be trained for your custom task.
```

---

### 12\. **Using torchvision’s Functional Transforms for Data Augmentation 🎨**

**Boilerplate Code**:

```python
import torchvision.transforms.functional as F

augmented_image = F.hflip(image)  # Horizontally flip the image
```

**Use Case**: Apply specific transformations directly to images using functional transforms.

**Goal**: Perform custom augmentations (like flips, rotations) on images. 🎯

**Sample Code**:

```python
import torchvision.transforms.functional as F
from PIL import Image

image = Image.open("image.jpg")
augmented_image = F.hflip(image)  # Flip the image horizontally
augmented_image.show()
```

**Before Example**: You write custom code to apply specific image augmentations manually.

```bash
# Manually flipping an image:
image = image.transpose(Image.FLIP_LEFT_RIGHT)
```

**After Example**: With `torchvision`, functional transforms allow easy image augmentation for your dataset.

```bash
# Image is flipped horizontally using torchvision’s functional API.
```

---

### 13\. **Freezing and Unfreezing Layers in a Model for Training 🧊🔥**

**Boilerplate Code**:

```python
for param in model.features.parameters():
    param.requires_grad = False  # Freeze the feature extractor layers
```

**Use Case**: Control which layers of a model are trainable for fine-tuning or transfer learning.

**Goal**: Freeze or unfreeze specific layers of a pre-trained model to adjust training focus. 🎯

**Sample Code**:

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)

# Freeze the feature extractor layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final layer for training
model.fc = torch.nn.Linear(512, num_classes)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

**Before Example**: You train all layers of a model, requiring more time and computational power.

```bash
# Training the entire model without freezing layers:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**After Example**: Freezing base layers allows you to fine-tune only the final layers, reducing the time and computational resources required.

```bash
Base layers frozen, final layer updated for fine-tuning.
# Training is now more efficient with only the final layer being trainable.
```

---

### 14\. **Visualizing Model Feature Maps 🔍**

**Boilerplate Code**:

```python
activation = {}

def hook_fn(module, input, output):
    activation["feature_map"] = output

model.layer4[1].register_forward_hook(hook_fn)
```

**Use Case**: Visualize the feature maps of convolutional layers to better understand what a model "sees."

**Goal**: Use hooks to extract and visualize the feature maps of a model during forward passes. 🎯

**Sample Code**:

```python
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
activation = {}

def hook_fn(module, input, output):
    activation["feature_map"] = output

# Register a hook on layer4 of ResNet
model.layer4[1].register_forward_hook(hook_fn)

# Preprocess an image
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
image = Image.open("image.jpg")
image_tensor = transform(image).unsqueeze(0)

# Perform a forward pass
with torch.no_grad():
    model(image_tensor)

# Visualize the feature map
feature_map = activation["feature_map"]
print(feature_map.shape)
```

**Before Example**: You manually extract intermediate layers of a model, which can be tedious.

```bash
# Manually extracting intermediate layers:
features = model.layer4(image_tensor)
```

**After Example**: With hooks, you can automatically capture feature maps during a forward pass and visualize them.

```bash
torch.Size([1, 512, 7, 7])
# Feature map from the model's convolutional layers visualized.
```

---

### 15\. **Image Normalization for Training Models with torchvision 🧽**

**Boilerplate Code**:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Use Case**: Normalize image pixel values to improve model performance during training.

**Goal**: Standardize pixel intensity values across images to match pre-trained model expectations. 🎯

**Sample Code**:

```python
from torchvision import transforms
from PIL import Image

image = Image.open("image.jpg")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

normalized_image = transform(image)
print(normalized_image.mean())
```

**Before Example**: You manually normalize image pixel values, which is tedious and error-prone.

```bash
# Manually normalizing an image:
image = (image - mean) / std
```

**After Example**: `torchvision` applies normalization easily, standardizing the images for better training performance.

```bash
Normalized image ready for model input with mean around 0.
```

### 16\. **Using torchvision to Apply Random Crops 🌾**

**Boilerplate Code**:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
])
```

**Use Case**: Randomly crop parts of an image for data augmentation.

**Goal**: Introduce variability in the input data to improve model generalization by applying random cropping. 🎯

**Sample Code**:

```python
from torchvision import transforms
from PIL import Image

image = Image.open("image.jpg")
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
])

cropped_image = transform(image)
cropped_image.show()
```

**Before Example**: You manually crop images or write custom cropping code, which is inefficient.

```bash
# Manually cropping an image:
cropped_image = image.crop((0, 0, 224, 224))
```

**After Example**: `torchvision` provides an easy way to apply random crops as part of the data augmentation pipeline.

```bash
# Image randomly cropped and resized to 224x224.
```

---

### 17\. **Applying Random Horizontal Flips with torchvision ↔️**

**Boilerplate Code**:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
```

**Use Case**: Randomly flip images horizontally for data augmentation.

**Goal**: Add randomness to your training data by horizontally flipping images, improving model robustness. 🎯

**Sample Code**:

```python
from torchvision import transforms
from PIL import Image

image = Image.open("image.jpg")
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

flipped_image = transform(image)
flipped_image.show()
```

**Before Example**: You manually write custom functions to flip images for data augmentation.

```bash
# Manually flipping an image horizontally:
flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
```

**After Example**: With `torchvision`, horizontal flipping is automatically handled with a single function in the augmentation pipeline.

```bash
# Image randomly flipped horizontally with a 50% probability.
```

---

### 18\. **Converting PIL Images to PyTorch Tensors with torchvision 🖼️➡️📊**

**Boilerplate Code**:

```python
from torchvision import transforms

transform = transforms.ToTensor()
```

**Use Case**: Convert a PIL image to a PyTorch tensor to feed it into a model.

**Goal**: Prepare images for deep learning models by converting them from the PIL format to tensors. 🎯

**Sample Code**:

```python
from torchvision import transforms
from PIL import Image

image = Image.open("image.jpg")
transform = transforms.ToTensor()

image_tensor = transform(image)
print(image_tensor.shape)
```

**Before Example**: You manually convert pixel data to tensors using custom code.

```bash
# Manually converting an image to a tensor:
image_tensor = torch.tensor(image)
```

**After Example**: `torchvision` simplifies the process by automatically converting images to tensors.

```bash
torch.Size([3, 224, 224])
# Image converted to a tensor ready for input to a model.
```

---

### 19\. **Image Color Jittering with torchvision 🎨**

**Boilerplate Code**:

```python
from torchvision import transforms

transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
```

**Use Case**: Randomly adjust the brightness, contrast, saturation, and hue of an image for data augmentation.

**Goal**: Apply random color distortions to make models more robust to lighting conditions. 🎯

**Sample Code**:

```python
from torchvision import transforms
from PIL import Image

image = Image.open("image.jpg")
transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)

jittered_image = transform(image)
jittered_image.show()
```

**Before Example**: You manually adjust color properties using external libraries, requiring more code.

```bash
# Manually adjusting brightness or contrast:
image = ImageEnhance.Brightness(image).enhance(1.5)
```

**After Example**: `torchvision` automatically applies randomized color adjustments as part of the augmentation pipeline.

```bash
# Image displayed with randomized brightness, contrast, and hue adjustments.
```

---

### 20\. **Normalizing Image Data for Consistent Input to Models 🧽**

**Boilerplate Code**:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Use Case**: Normalize images by adjusting pixel values to match model expectations.

**Goal**: Standardize the pixel intensity values of images to improve model performance. 🎯

**Sample Code**:

```python
from torchvision import transforms
from PIL import Image

image = Image.open("image.jpg")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

normalized_image = transform(image)
print(normalized_image.mean())
```

**Before Example**: You manually normalize image pixel values using custom functions, which can be error-prone.

```bash
# Manually normalizing an image:
image = (image - mean) / std
```

**After Example**: `torchvision` provides an easy-to-use normalization method for efficient image preprocessing.

```bash
# Normalized image with standardized pixel values ready for model input.
```