
from common import *

from net.loss   import *
import net.lovasz_losses as L
from net.sync_batchnorm.batchnorm import *

from model.senet import SEResNeXtBottleneck, SENet


#########################################################################################

class ConvBn2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        #self.bn = nn.BatchNorm2d(out_channels)
        self.bn = SynchronizedBatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels ):
        super(Decoder, self).__init__()
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x ):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)#False
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        return x


###########################################################################################3

class UNetSEResNext50(nn.Module):

    def load_pretrain(self, pretrain_file):
        self.se_resnext.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self, loss_type='focal', depth_type=None ):
        super().__init__()
        self.loss_type = loss_type

        self.se_resnext = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                      dropout_p=None, inplanes=64, input_3x3=False,
                      downsample_kernel_size=1, downsample_padding=0,
                      num_classes=1000)

        self.encoder1 = self.se_resnext.layer0
        self.encoder2 = self.se_resnext.layer1
        self.encoder3 = self.se_resnext.layer2
        self.encoder4 = self.se_resnext.layer3
        self.encoder5 = self.se_resnext.layer4

        self.center = nn.Sequential(
            ConvBn2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(2048+256, 512, 64)
        self.decoder4 = Decoder(1024+ 64, 512, 64)
        self.decoder3 = Decoder( 512+ 64, 256, 64)
        self.decoder2 = Decoder( 256+ 64, 128, 64)
        self.decoder1 = Decoder(  64    ,  64, 64)

        self.fuse_pixel  = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d( 8,  1, kernel_size=1, padding=0),
        )

        self.fuse  = nn.Sequential(
            nn.Conv2d( 6,  1, kernel_size=1, padding=0),
        )

        self.logit_image = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        batch_size,C,H,W = x.shape

        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[2])/std[2],
            (x-mean[1])/std[1],
            (x-mean[0])/std[0],
        ],1)

        ## encoder
        e1 = self.encoder1(x )             #; print('e1',e1.size())
        e2 = self.encoder2(e1)             #; print('e2',e2.size())
        e3 = self.encoder3(e2)             #; print('e3',e3.size())
        e4 = self.encoder4(e3)             #; print('e4',e4.size())
        e5 = self.encoder5(e4)             #; print('e5',e5.size())

        ## binary classification
        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size,-1)   
        f = F.dropout(f, p=0.50, training=self.training)   
        logit_image = self.logit_image(f).view(-1)  

        ## center
        f = self.center(e5)                #; print('f',f.size())

        ## decoder
        d5 = self.decoder5(torch.cat([f,  e5], 1))  #; print('d5',f.size())
        d4 = self.decoder4(torch.cat([d5, e4], 1))  #; print('d4',f.size())
        d3 = self.decoder3(torch.cat([d4, e3], 1))  #; print('d3',f.size())
        d2 = self.decoder2(torch.cat([d3, e2], 1))  #; print('d2',f.size())
        d1 = self.decoder1(d2)   

        ## side output
        d2 = F.upsample(d2,scale_factor= 2, mode='bilinear',align_corners=False)
        d3 = F.upsample(d3,scale_factor= 4, mode='bilinear',align_corners=False)
        d4 = F.upsample(d4,scale_factor= 8, mode='bilinear',align_corners=False)
        d5 = F.upsample(d5,scale_factor=16, mode='bilinear',align_corners=False)

        f1 = self.fuse_pixel(d1)
        f2 = self.fuse_pixel(d2)
        f3 = self.fuse_pixel(d3)
        f4 = self.fuse_pixel(d4)
        f5 = self.fuse_pixel(d5)

        logit_pixel = [f1, f2, f3, f4, f5]

        ## fused output
        logit = self.fuse(torch.cat([ 
            f1, f2, f3, f4, f5,
            F.upsample(logit_image.view(batch_size,-1,1,1,), scale_factor=128, mode='nearest')
        ],1))

        return logit, logit_pixel, logit_image


    ##-----------------------------------------------------------------
    def criterion(self, logit, logit_pixel, logit_image, truth_pixel, is_average=True):
        
        ## image classification loss
        batch_size, c, h, w = truth_pixel.shape
        truth_image = torch.tensor(np.array([((truth_pixel[i, :, :, :].sum()) > 0.5)    for i in range(batch_size)]), dtype=torch.float32, device='cuda:0')
        loss_image = F.binary_cross_entropy_with_logits(logit_image, truth_image)

        ## segmentation loss
        k = len(logit_pixel)
        if self.loss_type == 'focal':
            loss_pixel = RobustFocalLoss2d()(logit_pixel[0], truth_pixel, type='sigmoid')
            for i in range(1, k):
                loss_pixel += RobustFocalLoss2d()(logit_pixel[i], truth_pixel, type='sigmoid')
            loss_pixel *= 1.0 / k
            loss_seg = RobustFocalLoss2d()(logit, truth_pixel, type='sigmoid')
        elif self.loss_type == 'lovasz':
            loss_pixel = L.lovasz_hinge(logit_pixel[0], truth_pixel)
            for i in range(1, k):
                loss_pixel += L.lovasz_hinge(logit_pixel[i], truth_pixel)
            loss_pixel *= 1.0 / k
            loss_seg = L.lovasz_hinge(logit, truth_pixel)

        ## non-empty image seg loss
        loss_pixel = loss_pixel * truth_image #loss for empty image is weighted 0
        # print(loss_pixel.size())

        if is_average:
            loss_pixel = loss_pixel.sum() / truth_image.sum()

        return loss_seg, loss_pixel, loss_image


    def metric(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        return dice


    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError



class UNetSENext154(nn.Module):

    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        self.senet154.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self, loss_type='focal', depth_type=None):
        super().__init__()

        self.loss_type = loss_type

        self.senet154 = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                dropout_p=0.2, num_classes=1000)

        self.conv1    = self.senet154.layer0
        self.encoder2 = self.senet154.layer1
        self.encoder3 = self.senet154.layer2
        self.encoder4 = self.senet154.layer3
        self.encoder5 = self.senet154.layer4

        self.center = nn.Sequential(
            ConvBn2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(2048+256, 512, 64)
        self.decoder4 = Decoder(1024+ 64, 512, 64)
        self.decoder3 = Decoder( 512+ 64, 256, 64)
        self.decoder2 = Decoder( 256+ 64, 128, 64)
        self.decoder1 = Decoder(  64    ,  64, 64)

        self.fuse_pixel  = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d( 8,  1, kernel_size=1, padding=0),
        )

        self.fuse  = nn.Sequential(
            nn.Conv2d( 6,  1, kernel_size=1, padding=0),
        )

        self.logit_image = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )



    def forward(self, x):
        batch_size,C,H,W = x.shape

        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[2])/std[2],
            (x-mean[1])/std[1],
            (x-mean[0])/std[0],
        ],1)


        ## encoder
        x  = self.conv1(x)      #; print('x',x.size())
        e2 = self.encoder2( x)  #; print('e2',e2.size())
        e3 = self.encoder3(e2)  #; print('e3',e3.size())
        e4 = self.encoder4(e3)  #; print('e4',e4.size())
        e5 = self.encoder5(e4)  #; print('e5',e5.size())

        ## empty/non-empty binary classification loss
        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size,-1)   
        f = F.dropout(f, p=0.50, training=self.training)   
        logit_image = self.logit_image(f).view(-1)   

        ## center
        f = self.center(e5)
        
        ## side output
        d5 = self.decoder5(torch.cat([f,  e5], 1))  #; print('d5',f.size())
        d4 = self.decoder4(torch.cat([d5, e4], 1))  #; print('d4',f.size())
        d3 = self.decoder3(torch.cat([d4, e3], 1))  #; print('d3',f.size())
        d2 = self.decoder2(torch.cat([d3, e2], 1))  #; print('d2',f.size())
        d1 = self.decoder1(d2)                      #; print('d1',f.size())

        d2 = F.upsample(d2,scale_factor= 2, mode='bilinear',align_corners=False)
        d3 = F.upsample(d3,scale_factor= 4, mode='bilinear',align_corners=False)
        d4 = F.upsample(d4,scale_factor= 8, mode='bilinear',align_corners=False)
        d5 = F.upsample(d5,scale_factor=16, mode='bilinear',align_corners=False)

        f1 = self.fuse_pixel(d1)
        f2 = self.fuse_pixel(d2)
        f3 = self.fuse_pixel(d3)
        f4 = self.fuse_pixel(d4)
        f5 = self.fuse_pixel(d5)
        
        logit_pixel = [f1, f2, f3, f4, f5]

        ## fusion output
        logit = self.fuse(torch.cat([ 
            f1, f2, f3, f4, f5,
            F.upsample(logit_image.view(batch_size,-1,1,1,), scale_factor=128, mode='nearest')
        ],1))

        return logit, logit_pixel, logit_image


    def criterion(self, logit, logit_pixel, logit_image, truth_pixel, is_average=True):
        
        ## image classification loss
        batch_size, c, h, w = truth_pixel.shape
        truth_image = torch.tensor(np.array([((truth_pixel[i, :, :, :].sum()) > 0.5)    for i in range(batch_size)]), dtype=torch.float32, device='cuda:0')
        loss_image = F.binary_cross_entropy_with_logits(logit_image, truth_image)

        ## segmentation loss
        k = len(logit_pixel)
        if self.loss_type == 'focal':
            loss_pixel = RobustFocalLoss2d()(logit_pixel[0], truth_pixel, type='sigmoid')
            for i in range(1, k):
                loss_pixel += RobustFocalLoss2d()(logit_pixel[i], truth_pixel, type='sigmoid')
            loss_pixel *= 1.0 / k
            loss_seg = RobustFocalLoss2d()(logit, truth_pixel, type='sigmoid')
        elif self.loss_type == 'lovasz':
            loss_pixel = L.lovasz_hinge(logit_pixel[0], truth_pixel)
            for i in range(1, k):
                loss_pixel += L.lovasz_hinge(logit_pixel[i], truth_pixel)
            loss_pixel *= 1.0 / k
            loss_seg = L.lovasz_hinge(logit, truth_pixel)

        ## non-empty image seg loss
        loss_pixel = loss_pixel * truth_image #loss for empty image is weighted 0

        if is_average:
            loss_pixel = loss_pixel.sum() / truth_image.sum()

        return loss_seg, loss_pixel, loss_image


    def metric(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        return dice


    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


### run ##############################################################################

def run_check_net(loss_type='focal', network='se_resnext50'):

    batch_size = 8
    C,H,W = 1, 128, 128

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (2,   (batch_size,C,H,W)).astype(np.float32)

    truth_image = np.array([1, 1, 1,1,1,1,1,1]).astype(np.float32) 

    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).float().cuda()
    truth_image = torch.from_numpy(truth_image).float().cuda()

    #------------
    if network == 'se_resnext50':
        net = UNetSEResNext50(loss_type=loss_type).cuda()
        net.set_mode('train')
        net.load_pretrain(DATA_DIR + '/model/se_resnext50_32x4d-a260b3a4.pth')
    elif network = 'senet154':
        net = UNetSENext154(loss_type=loss_type).cuda()
        net.set_mode('train')
        net.load_pretrain(DATA_DIR + '/model/senet154-c7b49a05.pth')

    logit, logit_pixel, logit_image = net(input)
    loss_seg, loss_pixel, loss_image = net.criterion(logit, logit_pixel, logit_image, truth, is_average=True)
    loss  = loss_seg + 0.1 * loss_pixel + 0.05 * loss_image
    dice  = net.metric(logit, truth)

    print('loss_seg: %.8f,    loss_pixel: %.8f,    loss_image: %.8f' % (loss_seg.item(), loss_pixel.item(), loss_image.item()))
    print('dice : %0.8f'%dice.item())
    print('')

    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(net.parameters(), lr=0.001)

    i=0
    optimizer.zero_grad()
    while i<=1000:

        logit, logit_pixel, logit_image = net(input)
        loss_seg, loss_pixel, loss_image = net.criterion(logit, logit_pixel, logit_image, truth, is_average=True)
        loss  = loss_seg + 0.1 * loss_pixel + 0.05 * loss_image
        dice  = net.metric(logit, truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%20==0:
            print('loss_seg: %.8f,    loss_pixel: %.8f,    loss_image: %.8f' % (loss_seg.item(), loss_pixel.item(), loss_image.item()))


            print('[%05d] loss, dice  :  %0.5f,%0.5f'%(i, loss.item(),dice.item()))
        i = i+1


########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print( 'sucessful!')