import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def UpBlock(in_ch, out_ch, dropout=0.0):
    # TODO: try other upscaling blocks
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_ch, out_ch*2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_ch*2),
        nn.GLU(dim=1),
        nn.Dropout2d(dropout),
    )


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        nc = config['nc']
        nz = config['nz']
        ngf = config['ngf']

        dropout = config['g_dropout']

        nfc_base = {4:16, 8:8, 16:4, 32:2, 64:1}
        nfc = {k:int(v*ngf) for k,v in nfc_base.items()}

        self.init = nn.Sequential(
            nn.ConvTranspose2d(nz, nfc[4]*2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nfc[4]*2),
            nn.GLU(dim=1),
            nn.Dropout2d(dropout),
            # nn.SiLU(inplace=True),
            
#             nn.ConvTranspose2d(nfc[4], nfc[4], 3, 1, 1, bias=False),
#             nn.BatchNorm2d(nfc[4]),
#             nn.GLU(dim=1),
            # nn.ConvTranspose2d(nz, nfc[4], 4, 1, 0, bias=False),
            # nn.BatchNorm2d(nfc[4]),
            # nn.ReLU(),
            # nn.Dropout2d(dropout),

            # nn.ConvTranspose2d(nfc[4], nfc[8], 4, 2, 1, bias=False),
            # nn.BatchNorm2d(nfc[8]),
            # nn.ReLU(),
            # nn.Dropout2d(dropout),
        )

        self.feat_8 = UpBlock(nfc[4], nfc[8], dropout)
        self.feat_16 = UpBlock(nfc[8], nfc[16], dropout)
        self.feat_32 = UpBlock(nfc[16], nfc[32], dropout)
        self.feat_64 = UpBlock(nfc[32], nfc[64], dropout)
        # self.feat_64 = UpBlock(nfc[32], nc)

        self.to_64 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(nfc[32], nc, 3, 1, 1, bias=False),
            nn.Conv2d(nfc[64], nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)

        feat_4 = self.init(noise)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)
        feat_64 = self.feat_64(feat_32)

        return self.to_64(feat_64)


class GeneratorOld(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

        nz = config['nz']
        ngf = config['ngf']
        dropout = 0.2 # config['dropout_D']
        '''
        self.main = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*16, momentum=BN_momentum),
            nn.GLU(dim=1),
#             nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*8, ngf*8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf*8, momentum=BN_momentum),
            nn.GLU(dim=1),
#             nn.ReLU(),
            nn.Dropout2d(dropout),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*4, ngf*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf*4, momentum=BN_momentum),
            nn.GLU(dim=1),
#             nn.ReLU(),
            nn.Dropout2d(dropout),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*2, ngf*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf*2, momentum=BN_momentum),
            nn.GLU(dim=1),
#             nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )
        '''
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8, ),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            # nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf*8, ),
            # nn.ReLU(),
            # nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4,),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2, ),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf, ),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.main.apply(weights_init)
    
    def forward(self, x):
        x = x.view(-1, self.config['nz'], 1, 1)
        return self.main(x)


def DownBlock(in_planes, out_planes, dropout=0.0):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout2d(dropout),
    )


# TODO: train as encoder
#  - try both perceptual loss and pixel loss

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        nc = config['nc']
        ndf = config['ndf']

        dropout = config['d_dropout']

        nfc_base = {2:32, 4:16, 8:8, 16:4, 32:2}
        nfc = {k:int(v*ndf) for k,v in nfc_base.items()}

        self.start_block = nn.Sequential(
            nn.Conv2d(nc, nfc[32], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )

        self.down_to_16 = DownBlock(nfc[32], nfc[16], dropout)
        self.down_to_8 = DownBlock(nfc[16], nfc[8], dropout)
        self.down_to_4 = DownBlock(nfc[8], nfc[4], dropout)

        self.rf_main = nn.Sequential(
            nn.Conv2d(nfc[4], 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, image):
        feat_32 = self.start_block(image)
        feat_16 = self.down_to_16(feat_32)
        feat_8 = self.down_to_8(feat_16)
        feat_4 = self.down_to_4(feat_8)

        rf = self.rf_main(feat_4)

        return rf


class DiscriminatorOld(nn.Module):
    def __init__(self, config):
    # def __init__(self, nz=128, ndf=32, dropout=0.2, leakyReLU_slope=0.2,
                #  noisy_input=0.2, BN_momentum=0.2):
        super(Discriminator, self).__init__()
        self.config = config

        ndf = config['ndf']
        
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4, ),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8, ),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),
        )
        self.main.apply(weights_init)
    
    def forward(self, x):
        return self.main(x)


"""
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

        nz = config['nz']
        ngf = config['ngf']
        dropout = config['dropout_D']
        BN_momentum = config['BN_momentum']

        self.main = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*16, momentum=BN_momentum),
            nn.GLU(dim=1),
#             nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*8, ngf*8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf*8, momentum=BN_momentum),
            nn.GLU(dim=1),
#             nn.ReLU(),
            nn.Dropout2d(dropout),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*4, ngf*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf*4, momentum=BN_momentum),
            nn.GLU(dim=1),
#             nn.ReLU(),
            nn.Dropout2d(dropout),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*2, ngf*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf*2, momentum=BN_momentum),
            nn.GLU(dim=1),
#             nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )
        '''
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8, momentum=BN_momentum),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            # nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf*8, momentum=BN_momentum),
            # nn.ReLU(),
            # nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4, momentum=BN_momentum),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2, momentum=BN_momentum),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf, momentum=BN_momentum),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        '''
        self.main.apply(weights_init)
    
    def forward(self, x):
        x = x.view(-1, self.config['nz'], 1, 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, config):
    # def __init__(self, nz=128, ndf=32, dropout=0.2, leakyReLU_slope=0.2,
                #  noisy_input=0.2, BN_momentum=0.2):
        super(Discriminator, self).__init__()
        self.config = config
        # self.noisy_input = noisy_input

        nz = config['nz']
        ndf = config['ndf']
        dropout = config['dropout_D']
        BN_momentum = config['BN_momentum']
        leakyReLU_slope = config['leakyReLU_slope']
        
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(leakyReLU_slope),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2, momentum=BN_momentum),
            nn.LeakyReLU(leakyReLU_slope),
            nn.Dropout2d(dropout),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4, momentum=BN_momentum),
            nn.LeakyReLU(leakyReLU_slope),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8, momentum=BN_momentum),
            nn.LeakyReLU(leakyReLU_slope),
            nn.Dropout2d(dropout),
            
#             nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf*16, momentum=BN_momentum),
#             nn.LeakyReLU(leakyReLU_slope),
#             nn.Dropout2d(dropout),
            
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),
        )
        self.main.apply(weights_init)
    
    def forward(self, x):
        if self.training and self.config['noisy_input_D'] > 0:
            x = x + self.config['noisy_input_D']*torch.randn_like(x)

        return self.main(x)
"""
