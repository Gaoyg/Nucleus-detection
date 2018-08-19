from common import *
from models.loss import *
import data.dataset

if __name__ == '__main__':
    from configuration import *
else:
    from .configuration import *


## block ######################################################################################################

class ConvBn2d(nn.Module):

    def merge_bn(self):
        #raise NotImplementedError
        assert(self.conv.bias==None)
        conv_weight     = self.conv.weight.data
        bn_weight       = self.bn.weight.data
        bn_bias         = self.bn.bias.data
        bn_running_mean = self.bn.running_mean
        bn_running_var  = self.bn.running_var
        bn_eps          = self.bn.eps

        #https://github.com/sanghoon/pva-faster-rcnn/issues/5
        #https://github.com/sanghoon/pva-faster-rcnn/commit/39570aab8c6513f0e76e5ab5dba8dfbf63e9c68c

        N,C,KH,KW = conv_weight.size()
        std = 1/(torch.sqrt(bn_running_var+bn_eps))
        std_bn_weight =(std*bn_weight).repeat(C*KH*KW,1).t().contiguous().view(N,C,KH,KW )
        conv_weight_hat = std_bn_weight*conv_weight
        conv_bias_hat   = (bn_bias - bn_weight*std*bn_running_mean)

        self.bn   = None
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                              padding=self.conv.padding, stride=self.conv.stride, dilation=self.conv.dilation, groups=self.conv.groups,
                              bias=True)
        self.conv.weight.data = conv_weight_hat #fill in
        self.conv.bias.data   = conv_bias_hat



    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels, eps=1e-5)

        if is_bn is False:
            self.bn =None

    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class SEScale(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEScale, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x,1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x



#############  resnext50 pyramid feature net ###################################################################
# https://github.com/Hsuxu/ResNeXt/blob/master/models.py
# https://github.com/D-X-Y/ResNeXt-DenseNet/blob/master/models/resnext.py
# https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py


## P layers ## ---------------------------
class LateralBlock(nn.Module):
    def __init__(self, c_planes, p_planes, out_planes ):
        super(LateralBlock, self).__init__()
        self.lateral = nn.Conv2d(c_planes,  p_planes,   kernel_size=1, padding=0, stride=1)
        self.top     = nn.Conv2d(p_planes,  out_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, c , p):
        _,_,H,W = c.size()
        c = self.lateral(c)
        p = F.upsample(p, scale_factor=2, mode='nearest')
        p = p[:,:,:H,:W] + c
        p = self.top(p)

        return p

## C layers ## ---------------------------

# bottleneck type C
class SENextBottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, out_planes, groups, reduction=16, is_downsample=False, stride=1):
        super(SENextBottleneckBlock, self).__init__()
        self.is_downsample = is_downsample

        self.conv_bn1 = ConvBn2d(in_planes,     planes, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(   planes,     planes, kernel_size=3, padding=1, stride=stride, groups=groups)
        self.conv_bn3 = ConvBn2d(   planes, out_planes, kernel_size=1, padding=0, stride=1)
        self.scale    = SEScale(out_planes, reduction)


        if is_downsample:
            self.downsample = ConvBn2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):

        z = F.relu(self.conv_bn1(x),inplace=True)
        z = F.relu(self.conv_bn2(z),inplace=True)
        z =        self.conv_bn3(z)

        if self.is_downsample:
            z = self.scale(z)*z + self.downsample(x)
        else:
            z = self.scale(z)*z + x

        z = F.relu(z,inplace=True)
        return z



def make_layer_c0(in_planes, out_planes):
    layers = [
        nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)



def make_layer_c(in_planes, planes, out_planes, groups, num_blocks, stride):
    layers = []
    layers.append(SENextBottleneckBlock(in_planes, planes, out_planes, groups, is_downsample=True, stride=stride))
    for i in range(1, num_blocks):
        layers.append(SENextBottleneckBlock(out_planes, planes, out_planes, groups))

    return nn.Sequential(*layers)


class predict_fpn_block(nn.Module):
    def __init__(self, in_planes, out_planes, upsample=None ):
        super(predict_fpn_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,  out_planes,   kernel_size=3, padding=1, stride=1)
        self.conv2     = nn.Conv2d(out_planes,  out_planes, kernel_size=3, padding=1, stride=1)

        self.upsample = upsample

    def forward(self, x ):
        z = F.relu(self.conv1(x), inplace=True)
        z = F.relu(self.conv2(z), inplace=True)
        if self.upsample:
            z = F.upsample(z, scale_factor=self.upsample, mode='nearest')

        return z


#resnext50_32x4d
class FeatureNet(nn.Module):

    def __init__(self, cfg, in_channels, mid_channels=256, out_channels=128 ):
        super(FeatureNet, self).__init__()
        self.cfg=cfg

        # bottom-top
        self.layer_c0 = make_layer_c0(in_channels, 64)


        self.layer_c1 = make_layer_c(   64,  64,  256, groups=32, num_blocks=3, stride=1)  #out =  64*4 =  256
        self.layer_c2 = make_layer_c(  256, 128,  512, groups=32, num_blocks=4, stride=2)  #out = 128*4 =  512
        self.layer_c3 = make_layer_c(  512, 256, 1024, groups=32, num_blocks=6, stride=2)  #out = 256*4 = 1024
        self.layer_c4 = make_layer_c( 1024, 512, 2048, groups=32, num_blocks=3, stride=2)  #out = 512*4 = 2048


        # top-down
        self.layer_p4 = nn.Conv2d   ( 2048, mid_channels, kernel_size=1, stride=1, padding=0)
        self.layer_p3 = LateralBlock( 1024, mid_channels, mid_channels)
        self.layer_p2 = LateralBlock(  512, mid_channels, mid_channels)
        self.layer_p1 = LateralBlock(  256, mid_channels, mid_channels)

        self.pred_p4 = predict_fpn_block(mid_channels, out_channels, 8)
        self.pred_p3 = predict_fpn_block(mid_channels, out_channels, 4)
        self.pred_p2 = predict_fpn_block(mid_channels, out_channels, 2)
        self.pred_p1 = predict_fpn_block(mid_channels, out_channels)

    def forward(self, x):
        #pass                        #; print('input ',   x.size())
        c0 = self.layer_c0 (x)       #; print('layer_c0 ',c0.size())
                                     #
        c1 = self.layer_c1(c0)       #; print('layer_c1 ',c1.size())
        c2 = self.layer_c2(c1)       #; print('layer_c2 ',c2.size())
        c3 = self.layer_c3(c2)       #; print('layer_c3 ',c3.size())
        c4 = self.layer_c4(c3)       #; print('layer_c4 ',c4.size())

        p4 = self.layer_p4(c4)         #; print('layer_p4 ',p4.size())
        p3 = self.layer_p3(c3, p4)     #; print('layer_p3 ',p3.size())
        p2 = self.layer_p2(c2, p3)     #; print('layer_p2 ',p2.size())
        p1 = self.layer_p1(c1, p2)     #; print('layer_p1 ',p1.size())

        r4 = self.pred_p4(p4)          #; print('layer_r4', r4.size())
        r3 = self.pred_p3(p3)          #; print('layer_r3', r3.size())
        r2 = self.pred_p2(p2)          #; print('layer_r2', r2.size())
        r1 = self.pred_p1(p1)          #; print('layer_r1', r1.size())

        r = torch.cat([r4, r3, r2, r1], 1)    #; print(r.size())

        return r, c0 





############# various head ##############################################################################################
# Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
#   - Shi et. al (2016)


class FcnHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(FcnHead, self).__init__()

        #self.up   = nn.Upsample(scale_factor=2, mode='bilinear')
        #self.up   = nn.ConvTranspose2d(in_channels,in_channels, kernel_size=4, padding=1, stride=2, bias=False)

        self.conv1 = ConvBn2d(in_channels, 256, kernel_size=3, padding=1, stride=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
            # nn.Conv2d(512, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )

        self.logit = nn.Conv2d(64, 3, kernel_size=3, padding=1)


    # maxout: https://github.com/pytorch/pytorch/issues/805
    def forward(self, x):
        f = F.relu(self.conv1(x[0]), inplace=True)
        
        f = self.conv2(f)
        f = torch.cat([f, x[1]], 1)
        f = self.conv3(f)
        f = F.upsample(f, scale_factor=2, mode='nearest')
        f = self.conv4(f)
        f = F.dropout(f, p=0.5, training=self.training)
        logits = self.logit(f)

        return logits

#################### UNet ######################


class UNet(nn.Module):

    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.version = 'net version \'unet-se-resnext50-fpn\''
        self.cfg  = cfg
        self.mode = 'train'

        feature_channels = 128
        crop_channels = feature_channels
        self.feature_net = FeatureNet(cfg, 3, feature_channels*2, feature_channels)
        self.fcn_head    = FcnHead(cfg, feature_channels*4)

    def forward(self, inputs):
        cfg  = self.cfg
        mode = self.mode
        batch_size = len(inputs)

        #features
        features, c0    = data_parallel(self.feature_net, inputs)

        self.logits = data_parallel(self.fcn_head, [features, c0])




    def criterion(self, labels_truth):
        cfg  = self.cfg
        # logits = self.logits.permute(0, 2, 3, 1).contiguous().view(-1,C)
        # labels_truth = labels_truth.view(-1)
        # self.loss = F.cross_entropy(logits, labels_truth, size_average=True)

        self.loss = FocalLoss2d()(self.logits, labels_truth, class_weights=[1,1,3])


    #<todo> freeze bn for imagenet pretrain
    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)



# check #################################################################
def run_check_feature_net():

    batch_size = 2
    C, H, W = 3, 256, 256
    feature_channels = 128

    x = torch.randn(batch_size,C,H,W)
    inputs = Variable(x).cuda()

    cfg = Configuration()
    feature_net = FeatureNet(cfg, C, feature_channels).cuda()

    f = feature_net(inputs)

    print('')
    print( f.size())



def run_check_fcn_multi_head():

    batch_size  = 8
    in_channels = 128
    H,W = 256, 256

    h = int(H//2)
    w = int(W//2)
    f = np.random.uniform(-1,1,size=(batch_size,in_channels,h,w)).astype(np.float32)
    f = Variable(torch.from_numpy(f)).cuda()

    cfg = Configuration()
    fcn_head = FcnHead(cfg, in_channels).cuda()
    logits = fcn_head(f)

    print('logits ',logits.size())
    print('')

##-----------------------------------
def run_check_u_net():

    batch_size, C, H, W = 1, 3, 256,256
    feature_channels = 64
    inputs = np.random.uniform(-1,1,size=(batch_size, C, H, W)).astype(np.float32)
    inputs = Variable(torch.from_numpy(inputs)).cuda()

    cfg = Configuration()
    net = UNet(cfg).cuda()

    net.set_mode('eval')
    net(inputs)


    print('logits  ',net.logits.size())
    print('')




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # run_check_feature_net()
    # run_check_fcn_multi_head()

    run_check_u_net()