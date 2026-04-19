import torch 
import torch.nn as nn 

# Architecture from the original paper
# padding calculated by hand 
architecture_config = [
    # Kernel_size, n_filter_as_output, stride, padding
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    # parameters as before + number of repetitions 
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):   
        x = self.conv(x)        # apply the filters
        x = self.batchnorm(x)   # normalize the output
        x = self.leakyrelu(x)   # non-linearity
        return x
    
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        
    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1) # start_dim=1 to not flatten the number of examples
        x = self.fcs(x)
        return x
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if isinstance(x, tuple):
                layers += [ 
                    CNNBlock(
                        in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]
                # storing the output channels of the previous block as input channels to be used in the next iteration
                in_channels = x[1]
                
            elif isinstance(x, str):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
            elif isinstance(x, list):
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                
                for _ in range(num_repeats):
                    layers += [ 
                        CNNBlock(
                            in_channels, out_channels=conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3] 
                        )
                    ]
                    
                    layers += [ 
                        CNNBlock(                           
                            # the input channels of this CNNBlock is the output channels of the previous block
                            conv1[1], out_channels=conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3] 
                        )
                    ]
                # storing the output channels of the previous block as input channels to be used in the next iteration
                in_channels = conv2[1] 
        return nn.Sequential(*layers)
