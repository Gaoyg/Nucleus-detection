import numpy as np
import cv2
import os

def mask_merge(mask_sub_dir, H, W):

    mask = np.zeros((H, W), np.uint8)
    mask_img_list = [mask_img for mask_img in os.listdir(mask_sub_dir)]
    index = 0
    for mask_img in mask_img_list:
        index = index + 1
        sub_mask = cv2.imread(mask_sub_dir + mask_img, 0)
        sub_mask[sub_mask==255] = index
        mask |= sub_mask
  
    return mask

if __name__ == '__main__':

    root_dir = '/home/bio-eecs/gyg/nucleus_detection/data/val/'
    result_dir = '/home/bio-eecs/gyg/nucleus_detection/data/val_mask/'

    for img_name in os.listdir(root_dir): 
        img_dir = root_dir + img_name
        image = cv2.imread(img_dir+'/images/'+img_name+'.png', cv2.IMREAD_COLOR)

        H, W = image.shape[:2]
        
        mask_sub_dir = img_dir + '/masks/'

        mask_full = mask_merge(mask_sub_dir, H, W)

        cv2.imwrite(result_dir+img_name+'.png', mask_full)
    