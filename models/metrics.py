from common import *

def relabel_mask(mask):

    data = mask[:,:,np.newaxis]
    unique_color = set( tuple(v) for m in data for v in m )
    #print(len(unique_color))

    H,W  = data.shape[:2]
    mask = np.zeros((H,W),np.int32)
    for color in unique_color:
        #print(color)
        if color == (0,): continue

        m = (data==color).all(axis=2)
        label  = skimage.morphology.label(m)

        index = [label!=0]
        mask[index] = label[index]+mask.max()

    return mask



#https://www.kaggle.com/wcukierski/example-metric-implementation
def compute_precision(threshold, iou):
    matches = iou > threshold
    true_positives  = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

def print_precision(precision):

    print('thresh   prec    TP    FP    FN')
    print('---------------------------------')
    for (t, p, tp, fp, fn) in precision:
        print('%0.2f     %0.2f   %5.1f   %5.1f   %5.1f'%(t, p, tp, fp, fn))



def compute_average_precision_for_mask(predict, truth, t_range=np.arange(0.5, 1.0, 0.05)):
    predict = relabel_mask(predict)

    num_truth   = len(np.unique(truth  ))
    num_predict = len(np.unique(predict))

    # Compute intersection between all objects
    intersection = np.histogram2d(truth.flatten(), predict.flatten(), bins=(num_truth, num_predict))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(truth,   bins = num_truth  )[0]
    area_pred = np.histogram(predict, bins = num_predict)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred,  0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    precision = []
    average_precision = 0
    for t in t_range:
        tp, fp, fn = compute_precision(t, iou)
        p = tp / (tp + fp + fn)
        precision.append((t, p, tp, fp, fn))
        average_precision += p

    average_precision /= len(precision)
    return average_precision, precision

if __name__ == '__main__':
    
    mask_dir = ''
    predict_dir = '' 

    mask = cv2.imread(mask_dir, 0)
    predict = cv2.imread(predict_dir, 0)
        
    AP, precision = compute_average_precision_for_mask(predict, mask)