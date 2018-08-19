from common import *
from models.w_unet.unet_parts import *


class DNet(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DNet, self).__init__()

		self.version = 'net version \'WNet-DN\''
		
		self.inc = double_conv(in_channels, 64)
		self.down1 = down(64, 128)
		self.down2 = down(128, 128)
		self.down3 = down(128, 256)
		self.down4 = down(256, 256)
		self.down5 = down(256, 512)
		self.down6 = down(512, 512)

		self.up1 = up(1024, 256)
		self.up2 = up(512, 256)
		self.up3 = up(512, 128)
		self.up4 = up(256, 128)
		self.up5 = up(256, 64)
		self.up6 = up(128, 64)
		self.outc1 = outconv(64, out_channels)
		self.outc2 = outconv(64, out_channels)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x6 = self.down5(x5)
		x7 = self.down6(x6)

		z = self.up1(x7, x6)
		z = self.up2(z, x5)
		z = self.up3(z, x4)
		z = self.up4(z, x3)
		z = self.up5(z, x2)
		z = self.up6(z, x1)
		z1 = self.outc1(z)
		z2 = self.outc2(z)

		z1 = F.normalize(z1, p=2, dim=1)
		z2 = F.normalize(z2, p=2, dim=1)

		return z1, z2 

	def set_mode(self, mode ):
		self.mode = mode
		if mode in ['eval', 'valid', 'test']:
			self.eval()
		elif mode in ['train']:
			self.train()
		else:
			raise NotImplementedError


if __name__ == '__main__':

	x = torch.randn(1, 3, 256, 256)
	net = DNet(3, 2)

	output = net.forward(x)
	print(output.size())