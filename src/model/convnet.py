from torch import nn

# VGG16 architecture
class VGG16(nn.Module):
    def __init__(self, in_channels, n_classes, dropout):
        super(VGG16, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.drouput = dropout

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), # Conv 1-1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # Conv 1-2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Pool 1
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Conv 2-1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # Conv 2-2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Pool 2
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Conv 3-1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Conv 3-2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Conv 3-3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Pool 3
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # Conv 4-1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv 4-2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv 4-3
            nn.MaxPool2d(kernel_size=2, stride=2), # Pool 4
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv 5-1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv 5-2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv 5-3
            nn.MaxPool2d(kernel_size=2, stride=2), # Pool 5
        )
        
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),  # Average Pooling before FC layers
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),  # FC1
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 4096),  # FC2
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, n_classes),  # Output layer
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    



            
