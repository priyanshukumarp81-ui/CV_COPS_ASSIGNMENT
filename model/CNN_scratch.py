# %%
import torch
import torch.nn as nn

# %%
class convNeuralNet(nn.Module):
    def __init__(self,num_classes):
        super(convNeuralNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=7)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=7)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3)
        self.bn4 = nn.BatchNorm2d(256)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3)
        self.bn6 = nn.BatchNorm2d(512)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(512*22*22, 256)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.maxpool1(x)

        # Block 2
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.maxpool2(x)

        # Block 3
        x = self.activation(self.bn5(self.conv5(x)))
        x = self.activation(self.bn6(self.conv6(x)))
        x = self.maxpool3(x)
        
        # Pool & Flatten
        x = torch.flatten(x, 1) 

        # Dense Layers
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
            
            

import torch.nn.functional as F

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_tensor, target_class=None): 
        self.model.eval()

        with torch.enable_grad():
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
        
            self.model.zero_grad()
            score = output[0, target_class]
            score.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        weighted_activations = self.activations * pooled_gradients

        heatmap = torch.mean(weighted_activations, dim=1).squeeze(0)

        heatmap = F.relu(heatmap)
        
        heatmap = heatmap.detach().cpu()
        
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
            
        return heatmap.numpy(), target_class
   