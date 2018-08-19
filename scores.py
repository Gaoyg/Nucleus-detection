from common import *
from models.metrics import *

def mask_to_marker(mask):

    H,W = mask.shape[:2]
    yy,xx= np.mgrid[:H, :W]
    num_instances = mask.max()

    marker = np.zeros((H,W),np.int32)
    for i in range(num_instances):
        instance = mask==i+1
        distance = ndimage.distance_transform_edt(instance)
        distance = distance/distance.max()

        marker[distance>0.7] = i+1

    return marker


def label_to_mask(label):
    binary   = label > 0
    # marker = mask_to_marker(label==1)
    marker   = skimage.morphology.label(label==1)
    distance = -label
    water = skimage.morphology.watershed(-distance, marker, connectivity=1, mask=binary)

    return water

def get_score(predict_dir, mask_dir):
    sum_ap = 0
    count = 0
    for mask_name in os.listdir(mask_dir):
        index = mask_name[:-4]
        predict_name = index+'.png'
        mask = cv2.imread(mask_dir+mask_name, 0)
        # print (mask.shape)
        predict = cv2.imread(predict_dir+predict_name, 0)
        predict[predict== 120] = 1
        predict[predict== 240] = 2
        predict = label_to_mask(predict)
        # print(predict.shape)

        # predict = skimage.morphology.label(predict==1)
        
        AP, precision = compute_average_precision_for_mask(predict, mask)
        print(mask_name, AP)
        print_precision(precision)
        sum_ap += AP
        count += 1

    return sum_ap / count	

if __name__ == '__main__':
    predict_dir = '/home/bio-eecs/gyg/nucleus_detection/results/E2E_WNet-DN-6a/predict/E2E_WNet_DN_6a_val_identity/predicts/'
    # predict_dir = '/home/bio-eecs/gyg/nucleus_detection/results/unet-se-resnext50-single-1a/predict/single_label_1a_val_identity/predicts_1/'
    mask_dir = '/home/bio-eecs/gyg/nucleus_detection/data/val_mask/'

    map_res = get_score(predict_dir, mask_dir)
    print('MAP score:', map_res)
