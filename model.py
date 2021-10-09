import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class SEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Conv2d(in_ch, out_ch, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False),
            nn.Sigmoid(), # TODO: try others, like linear or tanh
        )

    def forward(self, small, large):
        #TODO: neutral param rename
        return large * self.main(small)


# TODO: try other upscaling blocks
def UpBlock(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_ch, out_ch*2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_ch*2),
        nn.GLU(dim=1),
    )

class UpBlockDual(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = UpBlock(in_ch, out_ch//2)
        self.dcgan_up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch//2),
            nn.ReLU(),
        )

    def forward(self, feat):
        a = self.main(feat)
        b = self.dcgan_up(feat)
        return torch.cat([a, b], dim=1)


class UpBlockSkip(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert in_ch == out_ch*2
        self.up =   nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch, out_ch*2, 3, 1, 1, bias=False)
        self.bn =   nn.BatchNorm2d(out_ch*2)
        self.glu =  nn.GLU(dim=1)

    def forward(self, feat):
        x = self.up(feat)
        y = self.conv(x)
        #y = self.bn(x + y)
        return self.glu(x + y)
        


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        nc = config['nc']
        nz = config['nz']
        ngf = config['ngf']

        nfc_base = {4:16, 8:8, 16:4, 32:2, 64:1}
        nfc = {k:int(v*ngf) for k,v in nfc_base.items()}

        self.init = nn.Sequential(
            nn.ConvTranspose2d(nz, nfc[4]*2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nfc[4]*2),
            nn.GLU(dim=1),
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

        self.up_block = {
            'UpBlock': UpBlock,
            'UpBlockSkip': UpBlockSkip,
            'UpBlockDual': UpBlockDual,
        }[config['g_up_block']]

        self.feat_8 = self.up_block(nfc[4], nfc[8])
        self.feat_16 = self.up_block(nfc[8], nfc[16])
        self.feat_32 = self.up_block(nfc[16], nfc[32])
        self.feat_64 = self.up_block(nfc[32], nfc[64])


        #self.se_32 = SEBlock(nfc[8], nfc[32])
        #self.se_64 = SEBlock(nfc[16], nfc[64])

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

        #feat_32 = self.se_32(feat_8, self.feat_32(feat_16))
        #feat_64 = self.se_64(feat_16, self.feat_64(feat_32))

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


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            noise = torch.randn_like(feat)

        return feat + self.weight * noise

def DownBlock(in_planes, out_planes, dropout=0.0):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        NoiseInjection(),
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
            NoiseInjection(),
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
        #if self.training and self.config['d_noise'] > 0:
            #x = x + self.config['d_noise']*torch.randn_like(x)

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
