from common import *

#函数功能：读取一幅图片的多个mask文件并叠加
def mask_read_and_stack(mask_sub_dir, H, W):
    mask = np.zeros((H, W), np.uint8)
    mask_img_list = [mask_img for mask_img in os.listdir(mask_sub_dir)]
    index = 0
    for mask_img in mask_img_list:
        index = index + 1
        sub_mask = cv2.imread(mask_sub_dir + mask_img, 0)
        sub_mask[sub_mask==255] = index
        mask |= sub_mask

    return mask

# Distance transform
def mask_to_distance(mask):

    H,W = mask.shape[:2]
    distance = np.zeros((H,W),np.float32)

    num_instances = mask.max()
    for i in range(num_instances):
        instance = mask==i+1
        d = ndimage.distance_transform_edt(instance)
        d = d/(d.max()+0.01)
        distance = distance+d

    distance = distance.astype(np.float32)
    return distance

# 前景
def mask_to_foreground(mask):
    foreground = (mask!=0)
    foreground = foreground.astype(np.int32)
    return foreground

# 找寻边界
def mask_to_border(mask):
    H,W = mask.shape[:2]
    border = np.zeros((H,W),np.float32)
    distance  = mask_to_distance(mask)
    y,x = np.where( np.logical_and(distance>0, distance<0.5) )
    border[y,x] = 1
    return border


# instances --> mask
def instance_to_mask(instance):
    H,W = instance.shape[1:3]
    mask = np.zeros((H,W),np.int32)

    num_instances = len(instance)
    for i in range(num_instances):
         mask[instance[i]>0] = i+1

    return mask

# mask --> instances
def mask_to_instance(mask):
    H,W = mask.shape[:2]
    num_instances = mask.max()
    instance = np.zeros((num_instances,H,W), np.float32)
    for i in range(num_instances):
         instance[i] = mask==i+1

    return instance

# 边界点为2，中心点为1
def mask_to_label(mask):
    H,W = mask.shape[:2]
    label    = np.zeros((H,W),np.float32)
    distance = mask_to_distance(mask)
    label[distance>0.5]=1  #center
    label[np.logical_and(distance>0,distance<=0.5)]=2  #boundary
    return label

# 找寻两个相交细胞的边界cut
def mask_to_cut(mask):
    H,W    = mask.shape[:2]
    dilate = np.zeros((H,W),np.bool)
    marker = np.zeros((H,W),np.bool)

    num_instances = mask.max()
    for i in range(num_instances):
        instance = mask==i+1
        d = ndimage.distance_transform_edt(instance)
        d_max = d.max()
        d = d/(d_max+1e-12)
        marker |= d>0.5

        radius = (instance.sum()/math.pi)**0.5
        r = 3 #int(round(max(1,radius*0.5)))
        dilate |= cv2.dilate(instance.astype(np.uint8),kernel=np.ones((r,r),np.float32)).astype(np.bool)

    marker   = skimage.morphology.label(marker)

    r = 5
    water1  = skimage.morphology.watershed(dilate, marker, mask=dilate)
    water2  = skimage.morphology.watershed(dilate, marker, mask=dilate, watershed_line=True)
    cut     = (water1-water2)!=0
    cut     = cv2.dilate(cut.astype(np.uint8),kernel=np.ones((r,r),np.float32)).astype(np.bool)

    return  cut

# 形成前景加边界的新的label
def mask_to_annotation(mask):
    label = mask_to_foreground(mask)
    cut = mask_to_cut(mask)
    label[np.where(cut)]=2
    return  label

def direction_from_border(mask):
    H,W    = mask.shape[:2]
    direction = np.zeros((2, H, W), np.int8)

    num_instances = mask.max()
    for i in range(num_instances+1):        # add background direction
        instance = (mask==i)
        edt, inds = ndimage.distance_transform_edt(instance, return_indices=True)
        border_vector = np.array([
                        np.expand_dims(np.arange(0, H), axis=1) - inds[0],
                        np.expand_dims(np.arange(0, W), axis=0) - inds[1]])
        
        direction[0] |= border_vector[0]
        direction[1] |= border_vector[1]
    
    direction_norm = direction / (np.linalg.norm(direction, axis=0, keepdims=True) + 1e-5)
    return direction_norm


def direction_to_center(mask):
    H, W = mask.shape[:2]
    num_instances = mask.max()
    direction = np.zeros((2, H, W))
    for i in range(num_instances+1):
        instance = (mask==i)
        center_of_mass = ndimage.measurements.center_of_mass(instance)
        current_offset_field = np.zeros((H, W, 2))
        current_offset_field[:, :, 0] = np.expand_dims(center_of_mass[0] - np.arange(0, H), axis=1)
        current_offset_field[:, :, 1] = np.expand_dims(center_of_mass[1] - np.arange(0, W), axis=0)
        direction[0][instance] = current_offset_field[:,:,0][instance]
        direction[1][instance] = current_offset_field[:,:,1][instance]
        
    direction_norm = direction / (np.linalg.norm(direction, axis=0, keepdims=True) + 1e-5)
    return direction_norm


def mask_to_weight(mask):
    H,W = mask.shape[:2]
    weight = np.zeros((H, W), np.float32)
    num_instances = mask.max()
    # print(num_instances)
    # print(mask)
    for i in range(num_instances+1):       # background as a big instance
        instance = (mask==i)
        area = np.sum(instance)
        weight[instance] = 1 / area
    
    weight = weight / (num_instances+1)

    return weight


def mask_to_center(mask):
    H, W = mask.shape[:2]
    num_instances = mask.max()
    center_mask = np.zeros((H, W), np.int32)
    for i in range(num_instances):
        instance = (mask==i+1)
        center = ndimage.measurements.center_of_mass(instance)
        center = np.round(center).astype(np.uint16)
        # print(i, center)
        center_mask[center[0]-1:center[0]+2, center[1]-1:center[1]+2] = 1

    return center_mask
    # H,W = mask.shape[:2]
    # label    = np.zeros((H,W),np.float32)
    # distance = mask_to_distance(mask)
    # label[distance>0.5]=1  #center

    # return label

# main function
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    mask = np.zeros((20, 20), np.int8)
    mask[5:15, 5:15] = 1
    label = mask_to_annotation(mask)
    io.imshow(label)

    print('\nsucess!')