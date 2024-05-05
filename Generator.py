import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, inp_vector, num_generated_features, num_channels, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.inp_vector = inp_vector
        self.num_generated_features = num_generated_features
        self.num_channels = num_channels
        self.model = nn.Sequential(
            # inp_vector * 1 * 1
            nn.ConvTranspose2d(inp_vector, num_generated_features*8, 4, 1, 0),
            nn.BatchNorm2d(num_generated_features*8),
            nn.ReLU(True),

            # (ngf*8) * 4 * 4
            nn.ConvTranspose2d(num_generated_features*8, num_generated_features*4, 4, 2, 1),
            nn.BatchNorm2d(num_generated_features*4),
            nn.ReLU(True),

            # (ngf*8) * 8 * 8
            nn.ConvTranspose2d(num_generated_features*4, num_generated_features*2, 4, 2, 1),
            nn.BatchNorm2d(num_generated_features*2),
            nn.ReLU(True),

            # (ngf*8) * 16 * 16
            nn.ConvTranspose2d(num_generated_features*2, num_generated_features, 4, 2, 1),
            nn.BatchNorm2d(num_generated_features),
            nn.ReLU(True),

            # (ngf*8) * 32 * 32
            nn.ConvTranspose2d(num_generated_features, num_channels, 4, 2, 1),
            nn.Tanh()

            # (ngf*8) * 64 * 64
        )

    def forward(self, vector):
        return self.model(vector)
    
