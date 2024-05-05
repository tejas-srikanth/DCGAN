import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random

import torchvision.datasets
import torch.utils.data

import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, ngpu, input_channels, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.input_channels = input_channels
        self.num_disc_features = ndf
        self.model = nn.Sequential(
            # Input 3*64*64
            nn.Conv2d(input_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            
            # (ndf*8)* 32 * 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),

            # (ndf*4) * 16 * 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            # (ndf*2) * 8 * 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),

            # (ndf) * 4 * 4
            nn.Conv2d(ndf*8, 1, 4, 1, 0),
            nn.Sigmoid()

            # 1 * 1 * 1
        )
    
    def forward(self, input):
        return self.model(input)
    
class Generator(nn.Module):
    def __init__(self, inp_vector, ngf, num_channels, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.inp_vector = inp_vector
        self.num_generated_features = ngf
        self.num_channels = num_channels
        self.model = nn.Sequential(
            # inp_vector * 1 * 1
            nn.ConvTranspose2d(inp_vector, ngf*8, 4, 1, 0),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            # (ngf*8) * 4 * 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # (ngf*8) * 8 * 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # (ngf*8) * 16 * 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf*8) * 32 * 32
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1),
            nn.Tanh()

            # 3 * 64 * 64
        )

    def forward(self, vector):
        return self.model(vector)

def init_weights(m):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(m.weight.data, 0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(args, discriminator, generator, dataloader, G_opt, D_opt, criterion, device, epoch, progress_file, loss_file):
    

    discriminator_losses = []
    generator_losses = []
    iters = 0
    real_label = 1
    fake_label = 0
    # grids = []

    # fake_noise = torch.randn(64, 100, 1, 1)

    for i, data in enumerate(dataloader):

        #TRAIN DISCRIMINATOR
        # run real images through discriminator
        images, _ = data
        images = images.to(device)
        batch_size = images.size()[0]
        labels_real = torch.full((batch_size,), real_label, dtype=torch.float, device=device)


        D_opt.zero_grad()
        output = discriminator(images).view(-1)
        dreal_loss = criterion(output, labels_real)
        dreal_loss.backward()

        # run fake images through discriminator
        start_vectors = torch.randn(batch_size, 100, 1, 1).to(device)
        g_output = generator(start_vectors)
        d_output = discriminator(g_output.detach()).view(-1)
        labels_fake = labels_real.fill_(fake_label)
        dfake_loss = criterion(d_output, labels_fake)
        dfake_loss.backward()

        #optimize discriminator
        dloss = (dfake_loss + dreal_loss)/2
        discriminator_losses.append(dloss.item())
        D_opt.step()

        #TRAIN GENERATOR
        G_opt.zero_grad()
        d_generated_images = discriminator(g_output).view(-1)
        labels_real = labels_real.fill_(real_label)
        dgen_loss = criterion(d_generated_images, labels_real)
        generator_losses.append(dgen_loss.item())
        dgen_loss.backward()
        G_opt.step()
          
        if (i % 50 == 0):
            progress_file.write(f'{epoch}, {i}, {len(dataloader)}, {dfake_loss.item()}, {dreal_loss.item()}, {dgen_loss.item()}')
            print(f'{epoch}, {i}, {len(dataloader)}, {dfake_loss.item()}, {dreal_loss.item()}, {dgen_loss.item()}')
  
        # if (iters % 500 == 0):
        #     with torch.no_grad():
        #         output = generator(fake_noise).detach()
        #         grids.append(output)

        loss_file.write(f'{dloss}, {dgen_loss}')

        iters += 1

def main():
    batch_size = 128
    nz = 100
    nc = 3
    ngf = 64
    ndf = 64
    image_size = 64
    num_workers = 2
    # Training settings
    parser = argparse.ArgumentParser(description='DCGAN THing')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta_1', type=float, default=0.5)

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(999)

    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.BCELoss()

    transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_folder = '/opt/ml/input/data/'

    dataset = dset.ImageFolder(root=data_folder,
                            transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    generator = Generator(nz, ngf, nc, 1).to(device)

    generator.apply(init_weights)

    discriminator = Discriminator(1, nc, ndf).to(device)

    discriminator.apply(init_weights)

    print(generator)
    print(discriminator)
    G_opt = optim.Adam(params=generator.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
    D_opt = optim.Adam(params=discriminator.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))

    discriminator.apply(init_weights)
    out_folder = '/opt/ml/output/'
    progress_file = open(out_folder + "progress.csv", "a")
    loss_file = open(out_folder + "losses.csv", "a")
    for i in range(1, args.epochs+1):
        train(args, discriminator, generator, dataloader, G_opt, D_opt, criterion, device, i, progress_file, loss_file)
    progress_file.close()
    loss_file.close()

    torch.save(discriminator.state_dict(), '/opt/ml/model/discriminator.pth')
    torch.save(generator.state_dict(), '/opt/ml/model/generator.pth')

if __name__ == '__main__':
    main()

    

    