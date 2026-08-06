import torch
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, n_classes):
        super(ResNet18, self).__init__()
        
        # Reduced dropout or remove completely inside residual blocks
        self.dropout_percentage = 0.1  
        self.relu = nn.ReLU()
        
        # BLOCK-1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
        
        # BLOCK-2
        self.conv2_1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2_1_1 = nn.BatchNorm2d(64)
        self.conv2_1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2_1_2 = nn.BatchNorm2d(64)
        self.dropout2_1 = nn.Dropout(p=self.dropout_percentage)
        
        self.conv2_2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2_2_1 = nn.BatchNorm2d(64)
        self.conv2_2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2_2_2 = nn.BatchNorm2d(64)
        self.dropout2_2 = nn.Dropout(p=self.dropout_percentage)
        
        # BLOCK-3
        self.conv3_1_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.batchnorm3_1_1 = nn.BatchNorm2d(128)
        self.conv3_1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3_1_2 = nn.BatchNorm2d(128)
        self.concat_adjust_3 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        self.batchnorm_adjust_3 = nn.BatchNorm2d(128)
        self.dropout3_1 = nn.Dropout(p=self.dropout_percentage)
        
        self.conv3_2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3_2_1 = nn.BatchNorm2d(128)
        self.conv3_2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3_2_2 = nn.BatchNorm2d(128)
        self.dropout3_2 = nn.Dropout(p=self.dropout_percentage)
        
        # BLOCK-4
        self.conv4_1_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.batchnorm4_1_1 = nn.BatchNorm2d(256)
        self.conv4_1_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4_1_2 = nn.BatchNorm2d(256)
        self.concat_adjust_4 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)
        self.batchnorm_adjust_4 = nn.BatchNorm2d(256)
        self.dropout4_1 = nn.Dropout(p=self.dropout_percentage)
        
        self.conv4_2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4_2_1 = nn.BatchNorm2d(256)
        self.conv4_2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4_2_2 = nn.BatchNorm2d(256)
        self.dropout4_2 = nn.Dropout(p=self.dropout_percentage)
        
        # BLOCK-5
        self.conv5_1_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.batchnorm5_1_1 = nn.BatchNorm2d(512)
        self.conv5_1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm5_1_2 = nn.BatchNorm2d(512)
        self.concat_adjust_5 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)
        self.batchnorm_adjust_5 = nn.BatchNorm2d(512)
        self.dropout5_1 = nn.Dropout(p=self.dropout_percentage)
        
        self.conv5_2_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm5_2_1 = nn.BatchNorm2d(512)
        self.conv5_2_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm5_2_2 = nn.BatchNorm2d(512)
        self.dropout5_2 = nn.Dropout(p=self.dropout_percentage)
        
        # Final Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)
        self.out = nn.Linear(1000, n_classes)

    def forward(self, x):
        # Block 1
        x = self.relu(self.batchnorm1(self.conv1(x)))
        op1 = self.maxpool1(x)
        
        # Block 2
        x = self.relu(self.batchnorm2_1_1(self.conv2_1_1(op1)))
        x = self.batchnorm2_1_2(self.conv2_1_2(x))
        x = self.dropout2_1(x)
        op2_1 = self.relu(x + op1)

        x = self.relu(self.batchnorm2_2_1(self.conv2_2_1(op2_1)))
        x = self.batchnorm2_2_2(self.conv2_2_2(x))
        x = self.dropout2_2(x)
        op2 = self.relu(x + op2_1)
        
        # Block 3
        x = self.relu(self.batchnorm3_1_1(self.conv3_1_1(op2)))
        x = self.batchnorm3_1_2(self.conv3_1_2(x))
        x = self.dropout3_1(x)
        shortcut3 = self.batchnorm_adjust_3(self.concat_adjust_3(op2))
        op3_1 = self.relu(x + shortcut3)

        x = self.relu(self.batchnorm3_2_1(self.conv3_2_1(op3_1)))
        x = self.batchnorm3_2_2(self.conv3_2_2(x))
        x = self.dropout3_2(x)
        op3 = self.relu(x + op3_1)
        
        # Block 4
        x = self.relu(self.batchnorm4_1_1(self.conv4_1_1(op3)))
        x = self.batchnorm4_1_2(self.conv4_1_2(x))
        x = self.dropout4_1(x)
        shortcut4 = self.batchnorm_adjust_4(self.concat_adjust_4(op3))
        op4_1 = self.relu(x + shortcut4)

        x = self.relu(self.batchnorm4_2_1(self.conv4_2_1(op4_1)))
        x = self.batchnorm4_2_2(self.conv4_2_2(x))
        x = self.dropout4_2(x)
        op4 = self.relu(x + op4_1)

        # Block 5
        x = self.relu(self.batchnorm5_1_1(self.conv5_1_1(op4)))
        x = self.batchnorm5_1_2(self.conv5_1_2(x))
        x = self.dropout5_1(x)
        shortcut5 = self.batchnorm_adjust_5(self.concat_adjust_5(op4))
        op5_1 = self.relu(x + shortcut5)

        x = self.relu(self.batchnorm5_2_1(self.conv5_2_1(op5_1)))
        # FIX: Using conv5_2_2 and batchnorm5_2_2 here!
        x = self.batchnorm5_2_2(self.conv5_2_2(x))
        x = self.dropout5_2(x)
        op5 = self.relu(x + op5_1)

        # Final Layers
        x = self.avgpool(op5)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc(x))
        x = self.out(x)

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