import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, ngpu, input_channels, num_disc_features):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.input_channels = input_channels
        self.num_disc_features = num_disc_features
        self.model = nn.Sequential(
            # Input 3*64*64
            nn.Conv2d(input_channels, num_disc_features, 4, 2, 1),
            nn.BatchNorm2d(num_disc_features),
            nn.LeakyReLU(0.2, True),
            
            # (ndf*8)* 32 * 32
            nn.Conv2d(num_disc_features, num_disc_features*2, 4, 2, 1),
            nn.BatchNorm2d(num_disc_features*2),
            nn.LeakyReLU(0.2, True),

            # (ndf*4) * 16 * 16
            nn.Conv2d(num_disc_features*2, num_disc_features*4, 4, 2, 1),
            nn.BatchNorm2d(num_disc_features*4),
            nn.LeakyReLU(0.2, True),

            # (ndf*2) * 8 * 8
            nn.Conv2d(num_disc_features*4, num_disc_features*8, 4, 2, 1),
            nn.BatchNorm2d(num_disc_features*8),
            nn.LeakyReLU(0.2, True),

            # (ndf) * 4 * 4
            nn.Conv2d(num_disc_features*8, 1, 4, 1, 0),
            nn.Sigmoid()

            # 1 * 1 * 1
        )
    
    def forward(self, input):
        return self.model(input)