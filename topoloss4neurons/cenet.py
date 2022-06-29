import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from MedicalNet.models import cenet_resnet as resnet
from functools import partial

#import Constants

#nonlinearity = partial(F.relu, inplace=True)
nonlinearity = nn.ReLU(inplace=True)
device = 'cuda'

class opt:
    model = 'resnet'
    model_depth = 34
    n_seg_classes = 1 # We only segment neurons
    input_W = 64 #not used
    input_H = 64  #not used
    input_D = 64 #not used
    resnet_shortcut = 'B'
    no_cuda = False
    gpu_id = []
    phase = 'train'
    pretrain_path = '/cvlabdata1/home/zakariya/SegmentingBrains/codes/MedicalNet/pretrain/resnet_34_23dataset.pth'
    new_layer_names = []
    
def generate_model(opt):
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
    '''
        if not opt.no_cuda:
                if len(opt.gpu_id) > 1:
                    model = model.cuda() 
                    model = nn.DataParallel(model, device_ids=opt.gpu_id)
                    net_dict = model.state_dict() 
                else:
                    import os
                    os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
                    model = model.cuda() 
                    model = nn.DataParallel(model, device_ids=None)
                    net_dict = model.state_dict()
    '''  

    model = model.cuda() 
    net_dict = model.state_dict() 
    
    # load pretrain
    if opt.phase == 'train' and opt.pretrain_path:
        print ('loading pretrained model {}'.format(opt.pretrain_path))
        if str(opt.model_depth) not in opt.pretrain_path:
            raise Exception('Loaded wrong model number')
        pretrain = torch.load(opt.pretrain_path)
        #pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()} #doesnt load anything, arent the same
        pretrain_dict = {}
        for k, v in pretrain['state_dict'].items():
            for key in net_dict.keys():
                if key == k.replace('module.', ''):
                    pretrain_dict[key] = v
        
        
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict) ## pas sur

        new_parameters = [] 
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv3d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv3d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv3d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool3d(kernel_size=5, stride=5)
        self.pool4 = nn.MaxPool3d(kernel_size=6, stride=6)

        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        in_channels, h, w, d = x.size(1), x.size(2), x.size(3), x.size(4)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w, d), mode='nearest')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w, d), mode='nearest')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w, d), mode='nearest')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w, d), mode='nearest')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm3d(in_channels // 4)
        self.relu1 = nonlinearity
        
        #if not isLast:
        self.deconv2 = nn.ConvTranspose3d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm3d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv3d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm3d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CE_Net_(nn.Module):
    #def __init__(self, num_classes=Constants.BINARY_CLASS, num_channels=3):
    def __init__(self, num_channels=1):
        super(CE_Net_, self).__init__()

        filters = [64, 128, 256, 512]
        
        '''
        resnet = models.resnet34(pretrained=True).to('cuda')
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        '''
        
        #encoder
        pr = opt()
        model, params = generate_model(pr)
        self.down_path = model.to(device)
        
        #bottom
        self.dblock = DACblock(512).to(device)
        self.spp = SPPblock(512).to(device)

        #decoder
        self.decoder4 = DecoderBlock(516, filters[2]).to(device)
        self.decoder3 = DecoderBlock(filters[2], filters[1]).to(device)
        self.decoder2 = DecoderBlock(filters[1], filters[0]).to(device)
        self.decoder1 = DecoderBlock(filters[0], filters[0]).to(device)

        #self.finaldeconv1 = nn.ConvTranspose3d(filters[0], 32, 4, 2, 1).to(device)
        self.finaldeconv1 = nn.ConvTranspose3d(filters[0], 32, 1, 1).to(device)
        self.finalrelu1 = nonlinearity.to(device)
        self.finalconv2 = nn.Conv3d(32, 32, 3, padding=1).to(device)
        self.finalrelu2 = nonlinearity.to(device)
        # self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.finalconv3 = nn.Conv3d(32, 1, 3, padding=1).to(device)

    def forward(self, x):
        # Encoder
        '''
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        '''
        
        x1,x2,x3,x4,x5 = self.down_path(x)
        
        # Center
        
        #print("now dblock")
        e4 = self.dblock(x5)
        #print(e4.size())
        #print("now spp")
        e4 = self.spp(e4)
        #print(e4.size())
        
        # Decoder
        d4 = self.decoder4(e4) + x4
        #print(d4.size())
        d3 = self.decoder3(d4) + x3
        #print(d3.size())
        d2 = self.decoder2(d3) + x2
        #print(d2.size())
        d1 = self.decoder1(d2)
        #print(d1.size())
        
        #print('end of decoders')
        
        out = self.finaldeconv1(d1)
        #print(out.size())
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        #print(out.size())
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        #print(out.size())
        
        return F.sigmoid(out)
