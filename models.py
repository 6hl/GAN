import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, num_c=3, in_size=100, dim=64):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_size, dim * 8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(dim * 8)
        self.conv2 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim * 4)
        self.conv3 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim * 2)
        self.conv4 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(dim)
        self.conv5 = nn.ConvTranspose2d(dim, num_c, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.tanh(self.conv5(x))
        return x
    
class Discriminator(nn.Module):
    def __init__(self, num_c=3, dim=64, model=None):
        super(Discriminator, self).__init__()
        self.model = model
        self.conv1 = nn.Conv2d(num_c, dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim * 2)
        self.conv3 = nn.Conv2d(dim*2, dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim * 4)
        self.conv4 = nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(dim * 8)
        self.conv5 = nn.Conv2d(dim * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        if self.model == "DCGAN":
            x = torch.sigmoid(self.conv5(x))
        elif self.model == "WGAN":
            x = self.conv5(x)
        return x

class ACGenerator(nn.Module):
    ''' Implementation for cifar10 only: https://arxiv.org/pdf/1610.09585v4.pdf '''
    def __init__(self, num_c=3, in_size=110, dim=384):
        super(ACGenerator, self).__init__()
        #384, 768
        self.dim = int(dim)
        if dim != 384:
            final_dim = 64
        else:
            final_dim = int(dim/8)
        self.dense1 = nn.Linear(in_size, dim)
        self.conv1 = nn.ConvTranspose2d(dim, int(dim/2), kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(int(dim/2))
        self.conv2 = nn.ConvTranspose2d(int(dim/2), int(dim/4), kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(dim/4))
        self.conv3 = nn.ConvTranspose2d(int(dim/4), final_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(final_dim)
        self.conv4 = nn.ConvTranspose2d(final_dim, num_c, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = x.view(-1, 110)
        x = F.relu(self.dense1(x))
        x = F.relu(self.bn1(self.conv1(x.view(-1, self.dim, 1, 1))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.tanh(self.conv4(x))
        return x

class ACDiscriminator(nn.Module):
    def __init__(self, num_c=3, dim=16, class_num=10):
        super(ACDiscriminator, self).__init__()
        # 3, 16
        self.conv1 = nn.Conv2d(num_c, dim, kernel_size=3, stride=2, padding=1, bias=False)
        # 16, 32
        self.conv2 = nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim * 2)
        # 32, 64
        self.conv3 = nn.Conv2d(dim*2, dim*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim * 4)
        # 64, 128
        self.conv4 = nn.Conv2d(dim*4, dim*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(dim * 8)
        # 128, 256
        self.conv5 = nn.Conv2d(dim*8, dim*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(dim * 16)
        # 256, 512
        self.conv6 = nn.Conv2d(dim*16, dim*32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(dim * 32)
        # 4*4*512, 8*8*512
        self.dense1 = nn.Linear(4*4*512, 1)
        self.dense2 = nn.Linear(4*4*512, class_num)

    def forward(self, x):
        x = F.dropout(F.leaky_relu(self.conv1(x), 0.2))#, 0.5)
        x = F.dropout(F.leaky_relu(self.bn2(self.conv2(x)), 0.2))#, 0.5)
        x = F.dropout(F.leaky_relu(self.bn3(self.conv3(x)), 0.2))#, 0.5)
        x = F.dropout(F.leaky_relu(self.bn4(self.conv4(x)), 0.2))#, 0.5)
        x = F.dropout(F.leaky_relu(self.bn5(self.conv5(x)), 0.2))#, 0.5)
        x = F.dropout(F.leaky_relu(self.bn6(self.conv6(x)), 0.2))#, 0.5)
        x = x.view(-1, 4*4*512)
        tf = torch.sigmoid(self.dense1(x))
        pred = F.log_softmax(self.dense2(x), dim=1)

        # tf: true/false prediction, pred: predicted class
        return tf, pred