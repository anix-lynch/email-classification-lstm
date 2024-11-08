---
title: "Skip connection in Pytorch with CNN"
datePublished: Sun Oct 13 2024 13:08:17 GMT+0000 (Coordinated Universal Time)
cuid: cm27lpg5a00090aiffjwndne9
slug: skip-connection-in-pytorch-with-cnn
tags: ai, deep-learning, cnn, pytorch

---

In this code, you're setting up a **skip connection** in PyTorch using a simple convolutional neural network (CNN) with two convolutional layers. The skip connection allows information to "skip" one layer and be added back to the output of the next layer, helping prevent problems like the vanishing gradient.

Let’s break this down and provide a working version of the `forward()` function and an example of what the output might look like.

### Code Explanation

```python
import torch
import torch.nn as nn

seed = 172
torch.manual_seed(seed)

class SkipConnection(nn.Module):
    def __init__(self):
        super(SkipConnection, self).__init__()
        # First convolutional layer, input has 3 channels, output has 6 channels
        self.conv_layer1 = nn.Conv2d(3, 6, 2, stride=2, padding=2)
        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)
        # Second convolutional layer, input has 6 channels, output has 3 channels
        self.conv_layer2 = nn.Conv2d(6, 3, 2, stride=2, padding=2)
        # ReLU activation function for the second layer
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        # First forward pass through conv_layer1 and apply ReLU
        out1 = self.relu(self.conv_layer1(input))
        # Forward pass through conv_layer2 and apply ReLU
        out2 = self.relu2(self.conv_layer2(out1))
        # Skip connection: adding the input back to the output
        out = input + out2
        return out

# Instantiate the model
model = SkipConnection()

# Create a random input tensor (1 batch, 3 channels, 32x32 image)
input_tensor = torch.rand(1, 3, 32, 32)

# Forward pass through the model
output = model(input_tensor)

# Print the shapes of the input and output
print("Input shape: ", input_tensor.shape)
print("Output shape: ", output.shape)
```

### Code Walkthrough:

1. **Initialization (**`__init__`):
    
    * You create two convolutional layers:
        
        * `conv_layer1` takes a 3-channel input (like an RGB image) and produces a 6-channel output.
            
        * `conv_layer2` takes the 6-channel output and reduces it back to 3 channels.
            
    * You also apply **ReLU**, which introduces non-linearity.
        
2. **Forward Pass**:
    
    * The input passes through the first convolution layer and ReLU activation.
        
    * It then passes through the second convolution layer and another ReLU.
        
    * Finally, you add the **original input** to the output of the second layer to create a **skip connection**. This "skips" the second convolutional layer, allowing information to pass through more directly.
        

### Expected Output (Demo)

For an input of size `(1, 3, 32, 32)` (which simulates a batch of 1 RGB image of size 32x32):

```bash
Input shape:  torch.Size([1, 3, 32, 32])
Output shape:  torch.Size([1, 3, 32, 32])
```

* The input and output have the same shape, as the skip connection ensures that the structure remains the same. However, the output values have been transformed by the convolution and ReLU layers, with the skip connection preserving some of the original information.