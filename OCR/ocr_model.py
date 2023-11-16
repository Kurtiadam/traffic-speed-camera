import torch
import torch.nn as nn
from pytorch_model_summary import summary
import numpy as np
import random
import os
import sys
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, script_path)
from config_utils import load_config
import torch.nn.functional as F


config = load_config()



class LPReaderModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, num_classes=len(config["data"]["classes"]), loc_conv1_out_channels = 32,
                 loc_conv1_kernel_size = 7, loc_conv2_out_channels = 32, loc_conv2_kernel_size = 5, maxpool_kernel_size = 2, maxpool_stride = 2):
        super(LPReaderModel, self).__init__()


        self.view_size = loc_conv2_out_channels*int((((config["data"]["img_target_height"]-loc_conv1_kernel_size+1)/2)-loc_conv2_kernel_size+1)/2)*int((((config["data"]["img_target_width"]-loc_conv1_kernel_size+1)/2)-loc_conv2_kernel_size+1)/2)
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, loc_conv1_out_channels, kernel_size=loc_conv1_kernel_size),
            nn.MaxPool2d(maxpool_kernel_size, stride=maxpool_stride),
            nn.ReLU(True),
            nn.Conv2d(loc_conv1_out_channels, loc_conv2_out_channels, kernel_size=loc_conv2_kernel_size),
            nn.MaxPool2d(maxpool_kernel_size, stride=maxpool_stride),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(int(self.view_size), loc_conv1_out_channels),
            nn.ReLU(True),
            nn.Linear(loc_conv1_out_channels, 2 * 3)  # 2x3 affine matrix
        )

        self.custom_weights = torch.zeros_like(self.fc_loc[2].weight)
        self.custom_biases = torch.tensor([1, 0, 0,
                                           0, 1, 0], dtype=torch.float32)
        self.custom_matrix = self.custom_biases.reshape(1, 6)
        self.fc_loc[2].weight.data = self.custom_weights
        self.fc_loc[2].bias.data = self.custom_matrix


        self.l1 = self._conv_layer(in_channels, out_channels)
        self.l2 = self._conv_layer(out_channels * 2, out_channels * 2)
        self.l3 = self._conv_layer(out_channels * 4, out_channels * 4)
        self.l4 = self._conv_layer(out_channels * 8, out_channels * 8)
        self.l5 = self._conv_layer(out_channels * 16, out_channels * 16)

        self.classifier = nn.Conv2d(out_channels * 32, num_classes, kernel_size=1)

    def _conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, stride=2, padding=padding, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels*2)
        )

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, int(self.view_size))
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, ([x.size()[0], 3, 32, 320])) # N x C x H x W
        x = F.grid_sample(x, grid)

        return x


    def forward(self, x):
        x = self.stn(x)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        x = torch.squeeze(self.classifier(x))

        return x


def create_network(source=""):
    """
     Creates the network from a given model. If the source (.pth) is given, loads those weights into the model. Returns the model.
    """
    torch.backends.cudnn.deterministic = True
    random.seed(config["data"]["seed"])
    np.random.seed(config["data"]["seed"])
    torch.manual_seed(config["data"]["seed"])
    torch.cuda.manual_seed_all(config["data"]["seed"])

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    print(summary(LPReaderModel(), torch.zeros(64,3,config["data"]["img_target_height"],config["data"]["img_target_width"]), show_input=True))
    
    net = LPReaderModel()
    
    if len(source) != 0:
        net.load_state_dict(torch.load(source))
        net.to(device)

    net.to(device)

    return net

if __name__ == "__main__":
    print(summary(LPReaderModel(), torch.zeros(1,3,config["data"]["img_target_height"],config["data"]["img_target_width"]), show_input=True))
