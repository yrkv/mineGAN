import torch
import torch.nn as nn
import torch.nn.functional as F


nfc_base = {4:32, 8:32, 16:16, 32:16, 64:8, 128:4, 256:2, 512:1}

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


def UpBlock(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_ch, out_ch*2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_ch*2),
        nn.GLU(dim=1),
        #nn.Dropout2d(config['g_dropout']),
    )

def UpBlockDCGAN(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf*4,),
        nn.ReLU(),
        #nn.Dropout2d(config['g_dropout']),
    )

class UpBlockDual(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = UpBlock(in_ch, out_ch)
        self.up_dcgan = UpBlockDCGAN(in_ch, out_ch)

    def forward(self, feat):
        a = self.main(feat)
        b = self.up_dcgan(feat)
        return (a + b) / 2


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
        y = self.bn(y)
        return self.glu(x + y)
        


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        nc = config['nc']
        nz = config['nz']
        ngf = config['ngf']

        nfc = {k:int(v*ngf) for k,v in nfc_base.items()}

        self.init = nn.Sequential(
            nn.ConvTranspose2d(nz, nfc[4]*2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nfc[4]*2),
            nn.GLU(dim=1),
        )

        self.up_block = {
            'UpBlock': UpBlock,
            'UpBlockDCGAN': UpBlockDCGAN,
            'UpBlockSkip': UpBlockSkip,
            'UpBlockDual': UpBlockDual,
        }[config['g_up_block']]

        self.feat_8 = self.up_block(nfc[4], nfc[8])
        self.feat_16 = self.up_block(nfc[8], nfc[16])
        self.feat_32 = self.up_block(nfc[16], nfc[32])
        self.feat_64 = self.up_block(nfc[32], nfc[64])
        self.feat_128 = self.up_block(nfc[64], nfc[128])
        self.feat_256 = self.up_block(nfc[128], nfc[256])
        self.feat_512 = self.up_block(nfc[256], nfc[512])

        #self.se_64  = SEBlock(nfc[4], nfc[64])
        #self.se_128 = SEBlock(nfc[8], nfc[128])
        #self.se_256 = SEBlock(nfc[16], nfc[256])


        #self.se_32 = SEBlock(nfc[8], nfc[32])
        #self.se_64 = SEBlock(nfc[16], nfc[64])

        # self.feat_64 = UpBlock(nfc[32], nc)

        #self.to_64 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(nfc[32], nc, 3, 1, 1, bias=False),
            #nn.Conv2d(nfc[64], nc, 3, 1, 1, bias=False),
            #nn.Tanh(),
        #)

        self.to_512 = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='nearest'),
            #nn.Conv2d(nfc[128], nc, 3, 1, 1, bias=False),
            nn.Conv2d(nfc[512], nc, 3, 1, 1, bias=False),
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
        feat_128 = self.feat_128(feat_64)
        feat_256 = self.feat_256(feat_128)
        feat_512 = self.feat_512(feat_256)


        #feat_64 = self.se_64(feat_4, self.feat_64(feat_32))
        #feat_128 = self.se_128(feat_8, self.feat_128(feat_64))
        #feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

        #return self.to_64(feat_64)
        return self.to_512(feat_512)


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


def select_part(feat_size, part, ws=8):
    hw = feat_size - ws
    if hw == 0:
        return 0, 0

    y = part // hw % hw
    x = part % hw

    return y, x

def slice_small_part(small_t, k, part, ws=8):
    x, y = select_part(k, part, ws=ws)
    return small_t[:, :, y:y+ws, x:x+ws]


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        nc = config['nc']
        ndf = config['ndf']

        dropout = config['d_dropout']

        nfc = {k:int(v*ndf) for k,v in nfc_base.items()}

        self.start_block = nn.Sequential(
            nn.Conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )

        self.down_to_128 = DownBlock(nfc[256], nfc[128], dropout)
        self.down_to_64 = DownBlock(nfc[128], nfc[64], dropout)
        self.down_to_32 = DownBlock(nfc[64], nfc[32], dropout)
        self.down_to_16 = DownBlock(nfc[32], nfc[16], dropout)
        self.down_to_8 = DownBlock(nfc[16], nfc[8], dropout)
        #self.down_to_4 = DownBlock(nfc[8], nfc[4], dropout)

        self.rf_main = nn.Sequential(
            nn.Conv2d(nfc[8], 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

        if config['d_encoder'] > 0:
            self.decoder_512 = SimpleDecoder(nfc[8], nc, ndf=8) # 512
        if config['d_encoder'] > 1:
            self.decoder_256 = SimpleDecoder(nfc[16], nc, ndf=8) # 256
            self.decoder_128 = SimpleDecoder(nfc[32], nc, ndf=8) # 128
            self.decoder_64 = SimpleDecoder(nfc[64], nc, ndf=8) # 64

        self.apply(weights_init)


    def forward(self, image, label='fake', part=None):

        feat_256 = self.start_block(image)
        feat_128 = self.down_to_128(feat_256)
        feat_64 = self.down_to_64(feat_128)
        feat_32 = self.down_to_32(feat_64)
        feat_16 = self.down_to_16(feat_32)
        feat_8 = self.down_to_8(feat_16)

        rf = self.rf_main(feat_8)

        if label == 'real':
            rec = []
            if self.config['d_encoder'] > 0:
                rec.append(self.decoder_512(feat_8))

            if self.config['d_encoder'] > 1:
                assert part is not None
                rec.append(self.decoder_256(slice_small_part(feat_16, 16, part)))
                rec.append(self.decoder_128(slice_small_part(feat_32, 32, part)))
                rec.append(self.decoder_64(slice_small_part(feat_64, 64, part)))

            return rf, *rec

        return rf,

# TODO: train as encoder
#  - try both perceptual loss and pixel loss

class Discriminator_(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        nc = config['nc']
        ndf = config['ndf']

        dropout = config['d_dropout']

        nfc = {k:int(v*ndf) for k,v in nfc_base.items()}

        self.start_block = nn.Sequential(
            nn.Conv2d(nc, nfc[128], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )

        self.down_to_64 = DownBlock(nfc[128], nfc[64], dropout)
        self.down_to_32 = DownBlock(nfc[64], nfc[32], dropout)
        self.down_to_16 = DownBlock(nfc[32], nfc[16], dropout)
        self.down_to_8 = DownBlock(nfc[16], nfc[8], dropout)
        #self.down_to_4 = DownBlock(nfc[8], nfc[4], dropout)

        self.rf_main = nn.Sequential(
            nn.Conv2d(nfc[8], 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )


        self.down_from_small = nn.Sequential(
            nn.Conv2d(nc, nfc[32], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[32], nfc[16]),
            DownBlock(nfc[16], nfc[8]),
        )

        self.rf_small = nn.Sequential(
            nn.Conv2d(nfc[8], 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

        # decoder applied to end of main backbone
        self.decoder_big = SimpleDecoder(nfc[8], nc, ndf=ndf)
        # decoder applied to randomly selected part
        self.decoder_part = SimpleDecoder(nfc[16], nc, ndf=ndf)
        # decoder applied to simpler/smaller backbone
        self.decoder_small = SimpleDecoder(nfc[8], nc, ndf=ndf)

        self.apply(weights_init)


    def forward(self, image, label='fake', part=None):
        #if self.training and self.config['d_noise'] > 0:
            #x = x + self.config['d_noise']*torch.randn_like(x)

        feat_128 = self.start_block(image)
        feat_64 = self.down_to_64(feat_128)
        feat_32 = self.down_to_32(feat_64)
        feat_16 = self.down_to_16(feat_32)
        feat_8 = self.down_to_8(feat_16)
        #feat_4 = self.down_to_4(feat_8)

        rf_0 = self.rf_main(feat_8)

        small_image = F.interpolate(image, size=64)
        feat_small = self.down_from_small(small_image)
        rf_1 = self.rf_small(feat_small)

        rf = torch.cat([rf_0, rf_1], dim=1)

        if label == 'real':
            rec_img_big = self.decoder_big(feat_8)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            y_slice, x_slice = part
            rec_img_part = self.decoder_part(feat_16[:, :, y_slice, x_slice])

            return rf, rec_img_big, rec_img_small, rec_img_part

        return rf,


class SimpleDecoder(nn.Module):
    def __init__(self, nfc_in=64, nc=3, ndf=16):
        super(SimpleDecoder, self).__init__()

        nfc = {k:int(v*ndf) for k,v in nfc_base.items()}

        # def upBlock(in_planes, out_planes):
        #     return nn.Sequential(
        #         nn.Upsample(scale_factor=2, mode='nearest'),
        #         nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #         nn.BatchNorm2d(out_planes*2),
        #         nn.GLU(dim=1)

        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            UpBlock(nfc_in, nfc[16]),
            UpBlock(nfc[16], nfc[32]),
            UpBlock(nfc[32], nfc[64]),
            #UpBlock(nfc[64], nfc[128]),
            nn.Conv2d(nfc[64], nc, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)


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
