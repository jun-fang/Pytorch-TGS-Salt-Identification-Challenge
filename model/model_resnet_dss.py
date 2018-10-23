from common import *

# from net.loss   import *
import net.lovasz_losses as L
# from net.sync_batchnorm.batchnorm import *

from model.resnet import BasicBlock, ResNet


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        #self.bn = nn.BatchNorm2d(out_channels)
        self.bn = SynchronizedBatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, channels, out_channels, ):
        super(Decoder, self).__init__()
        self.conv1 =  ConvBn2d(in_channels_1 + in_channels_2,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2 ):
        if x1.size()[-2:] != x2.size()[-2:]:
            x1 = F.upsample(x1, scale_factor=2, mode='bilinear', align_corners=True)#False
        x = torch.cat([x1, x2], 1)
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        return x


class UNetResNet34(nn.Module):
# PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

	# def __init__(self ):
	# 	super(UNetResNet34,self).__init__()
	def __init__(self, loss_type='focal', depth_type=None):
		super().__init__()
		self.loss_type = loss_type

		self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1 )

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
			

		self.encoder1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
			nn.BatchNorm2d(64),
			# self.conv1,
			# self.bn1,
			nn.ReLU(inplace=True),
		)
		self.encoder2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			self.resnet.layer1,              # 64
		)
		self.encoder3 = self.resnet.layer2   # 128
		self.encoder4 = self.resnet.layer3   # 256
		self.encoder5 = self.resnet.layer4   # 512

		self.center = nn.Sequential(
			ConvBn2d( 512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			ConvBn2d( 512, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
		)

		self.decoder5 = Decoder(256, 512, 512, 64)
		self.decoder4 = Decoder( 64, 256, 256, 64)
		self.decoder3 = Decoder( 64, 128, 128, 64)
		self.decoder2 = Decoder( 64,  64,  64, 64)
		self.decoder1 = Decoder( 64,  64,  32, 64)

		self.fuse_pixel  = nn.Sequential(
			nn.Conv2d(64, 8, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d( 8,  1, kernel_size=1, padding=0),
		)

		self.fuse  = nn.Sequential(
			nn.Conv2d( 6,  1, kernel_size=1, padding=0),
		)

		self.logit_image = nn.Sequential(
			nn.Linear(512, 64),
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
		d5 = self.decoder5( f,e5)          #; print('d5',d5.size())
		d4 = self.decoder4(d5,e4)          #; print('d4',d4.size())
		d3 = self.decoder3(d4,e3)          #; print('d3',d3.size())
		d2 = self.decoder2(d3,e2)          #; print('d2',d2.size())
		d1 = self.decoder1(d2,e1)          #; print('d1',d1.size())

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


SaltNet = UNetResNet34


### run ##############################################################################

def run_check_net(loss_type='focal'):

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
	net = SaltNet(loss_type=loss_type).cuda()
	net.set_mode('train')

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