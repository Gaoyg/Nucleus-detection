from common import *
from data.preprocess import *

class ScienceDataset(Dataset):

	def __init__(self, root_path, split, transform=None, mode='train'):
		super(ScienceDataset, self).__init__()

		self.imgs_dir = root_path + split
		self.transform = transform
		self.mode = mode

		img_ids = [img for img in os.listdir(root_path+split)]
		self.img_ids = img_ids

	def __getitem__(self, index):
		id = self.img_ids[index]
		image = cv2.imread(self.imgs_dir + '/%s/images/%s.png'%(id, id), cv2.IMREAD_COLOR)

		H, W = image.shape[:2]
		if self.mode == 'train':
			mask = mask_read_and_stack(self.imgs_dir + '/%s/masks/'%(id), H, W)
			if self.transform is not None:
				return self.transform(image, mask, index)
			else:
				return image, mask, index

		if self.mode == 'test':
			if self.transform is not None:
				return self.transform(image, index)
			else:
				return image, index

	def __len__(self):
		return len(self.img_ids)