from common import *
from models.w_unet.unet_parts import *


class DNet(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DNet, self).__init__()

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
		f = self.up6(z, x1)
		z1 = self.outc1(f)
		z2 = self.outc2(f)

		z1 = F.normalize(z1, p=2, dim=1)
		z2 = F.normalize(z2, p=2, dim=1)

		return z1, z2, f 


class UNet2(nn.Module):
	def __init__(self):
		super(UNet2, self).__init__()

		self.inc = double_conv(68, 128)
		self.down1 = down(128, 256)
		self.down2 = down(256, 256)
		self.down3 = down(256, 512)
		self.down4 = down(512, 512)

		self.up1 = up(1024, 256)
		self.up2 = up(512, 256)
		self.up3 = up(512, 128)
		self.up4 = up(256, 128)

		self.center_conv = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1),
			nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1)
		)

		self.mask_conv = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1),
			nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1)
		)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)

		u = self.up1(x5, x4)
		u = self.up2(u, x3)
		u = self.up3(u, x2)
		u = self.up4(u, x1)

		u = F.dropout(u, p=0.5, training=self.training)

		center = self.center_conv(u)
		mask = self.mask_conv(u)

		return center, mask

class E2ENet(nn.Module):
	def __init__(self, in_channels, dn_out_channels):
		super(E2ENet, self).__init__()

		self.version = 'E2EWNet-DN+UNet'
		
		self.dnet = DNet(in_channels, dn_out_channels)
		self.unet2 = UNet2()

	def forward(self, x):
		dir_from_border_pred, dir_to_center_pred, features = data_parallel(self.dnet, x)
		x_in = torch.cat([dir_from_border_pred, dir_to_center_pred, features], dim=1)
		center, mask = data_parallel(self.unet2, x_in)

		return dir_from_border_pred, dir_to_center_pred, center, mask

	def set_mode(self, mode ):
		self.mode = mode
		if mode in ['eval', 'valid', 'test']:
			self.eval()
		elif mode in ['train']:
			self.train()
		else:
			raise NotImplementedError

	# pretrained direction net to initialize DN 
	def load_pretrain(self, pretrain_file, skip=[]):
		pretrain_state_dict = torch.load(pretrain_file)
		state_dict = self.dnet.state_dict()

		keys = list(state_dict.keys())
		for key in keys:
			if any(s in key for s in skip): continue
			state_dict[key] = pretrain_state_dict[key]

		self.dnet.load_state_dict(state_dict)


if __name__ == '__main__':

	x = torch.randn(1, 3, 256, 256)
	e2enet = E2ENet(3, 2)

	direction, center, mask = e2enet(x)

	print(direction.shape)
	print(center.shape)
	print(mask.shape)