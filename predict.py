import os, sys
sys.path.append(os.path.dirname(__file__))

#from train_multi_label import *
from train_E2EWNet import *

#--------------------------------------------------------------
def label_to_mask(label):
    binary   = label > 0
    marker   = skimage.morphology.label(label==1)
    distance = -label
    water = skimage.morphology.watershed(-distance, marker, connectivity=1, mask=binary)

    return water

#--------------------------------------------------------------
AUG_FACTOR = 64    # default 16,  64 for direction networks

def do_test_augment_identity(image):
    height,width = image.shape[:2]
    h = math.ceil(height/AUG_FACTOR)*AUG_FACTOR
    w = math.ceil(width /AUG_FACTOR)*AUG_FACTOR
    dx = w-width
    dy = h-height

    image = cv2.copyMakeBorder(image, left=0, top=0, right=dx, bottom=dy,
                               borderType= cv2.BORDER_REFLECT101, value=[0,0,0] )

    return image


def undo_test_augment_identity(net, image):
    height,width = image.shape[:2]

    prob  = np_softmax(net.logits.data.cpu().numpy())[0]
    label = np.argmax(prob,0).astype(np.float32)
    mask  = label_to_mask(label)

    mask = mask[:height, :width]
    return mask



def predict_one_image(image_name, net):
    image = cv2.imread(DATA_DIR + '/val/%s/images/%s.png'%(image_name,image_name), cv2.IMREAD_COLOR)
    augment_image  = do_test_augment_identity(image)

    net.set_mode('test')
    with torch.no_grad():
        input = torch.from_numpy(augment_image.transpose((2,0,1))).float().div(255).unsqueeze(0)
        input = Variable(input).cuda()
        net.forward(input)

    logits = net.logits.data.cpu().numpy()
    prob  = np_softmax(logits)[0]
    label = np.argmax(prob,0)

    return label
    


def run_predict():

    out_dir = RESULTS_DIR + '/E2E_WNet-DN-6a'
    initial_checkpoint = \
        RESULTS_DIR + 'E2E_WNet-DN-6a/checkpoint/00049999_model.pth'

    # augment -----------------------------------------------------------------------------------------------------
    augments=[
        #tag_name, do_test_augment, undo_test_augment, params
        ('identity', do_test_augment_identity, None, {}),
    ]

    #-----------------------------------------------------------------------------------
    split = '/val/'  #'valid1_ids_gray2_38'
    ids   = read_list_from_dir(DATA_DIR + split) #[:10] #try 10 images for debug

    #start experiments here! ###########################################################
    # os.makedirs(out_dir +'/backup', exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## net ------------------------------
    net = Net(3, 2).cuda()
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n'%(type(net)))
    log.write('\n')



    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')


    log.write('\ttsplit   = %s\n'%(split))
    log.write('\tlen(ids) = %d\n'%(len(ids)))
    log.write('\n')


    for tag_name, do_test_augment, undo_test_augment, params in augments:

        ## setup  --------------------------
        tag = 'E2E_WNet_DN_6a_val_%s'%tag_name   ##tag = 'test1_ids_gray2_53-00011000_model'
        # os.makedirs(out_dir +'/predict/%s/overlays'%tag, exist_ok=True)
        os.makedirs(out_dir +'/predict/%s/predicts'%tag, exist_ok=True)
        os.makedirs(out_dir +'/predict/%s/directions'%tag, exist_ok=True)

        # os.makedirs(out_dir +'/predict/%s/rcnn_proposals'%tag, exist_ok=True)
        # os.makedirs(out_dir +'/predict/%s/detections'%tag, exist_ok=True)
        # os.makedirs(out_dir +'/predict/%s/masks'%tag, exist_ok=True)
        # os.makedirs(out_dir +'/predict/%s/instances'%tag, exist_ok=True)



        log.write('** start evaluation here @%s! **\n'%tag)
        for i in range(0,len(ids)):


            name = ids[i]
            print('%03d %s'%(i,name))

            image = cv2.imread(DATA_DIR + '/val/%s/images/%s.png'%(name,name), cv2.IMREAD_COLOR)
            augment_image  = do_test_augment(image,  **params)


            net.set_mode('test')
            with torch.no_grad():
                input = torch.from_numpy(augment_image.transpose((2,0,1))).float().div(255).unsqueeze(0)
                input = Variable(input).cuda()
                dir_pred, dir1_pred, center_pred, mask_pred = net.forward(input)


            #----------------------------------------
            if 0:
                foregrounds = net.foregrounds.data.cpu().numpy()
                foreground = np_sigmoid(foregrounds)[0]
                cuts = net.cuts.data.cpu().numpy()
                cut = np_sigmoid(cuts)[0]

                foreground[np.where(foreground<=0.85)] = 0
                cut[np.where(cut<=0.5)] = 0


                height, width = image.shape[:2]
                H,W = augment_image.shape[:2]
                overlay_label = np.zeros((H,W,3), np.uint8)
                overlay_label[:,:,0]= foreground*255
                # overlay_label[:,:,1]= cut*255
                overlay_label = overlay_label[:height,:width]
                #image_show('label', overlay_label)

            if 0:
                logits = net.logits.data.cpu().numpy()
                prob  = np_softmax(logits)[0]
                label = np.argmax(prob,0)

                # prob[np.where(prob<0.5)]=0

                height, width = image.shape[:2]
                label = label[:height, :width]
                # H,W = augment_image.shape[:2]
                # overlay_label = np.zeros((H,W,3), np.uint8)
                # overlay_label[:,:,0]= prob[1,:,:]*255
                # overlay_label[:,:,1]= prob[2,:,:]*255
                # overlay_label = overlay_label[:height,:width]
                # image_show('label', overlay_label)

            if 1:
                height, width = image.shape[:2]
                res = np.zeros((height, width, 3), np.float32)
                dir_pred = dir_pred[0].data.cpu().numpy()
                dir_pred = dir_pred[:, :height, :width]
                # print(dir_pred)
                dir_y = dir_pred[0] + 1
                dir_x = dir_pred[1] + 1
                res[:,:,0] = dir_y * 128 
                res[:,:,1] = dir_x * 128 

                res1 = np.zeros((height, width, 3), np.float32)
                dir1_pred = dir1_pred[0].data.cpu().numpy()
                dir1_pred = dir1_pred[:, :height, :width]
                # print(dir_pred)
                dir1_y = dir1_pred[0] + 1
                dir1_x = dir1_pred[1] + 1
                res1[:,:,0] = dir1_y * 128 
                res1[:,:,1] = dir1_x * 128 

                center_pred = center_pred[0].data.cpu().numpy()
                mask_pred = mask_pred[0].data.cpu().numpy()

                center_pred = np_sigmoid(center_pred)[0][:height, :width]
                mask_pred = np_sigmoid(mask_pred)[0][:height, :width]

                center_pred[center_pred <= 0.85] = 0
                center_pred[center_pred > 0.85] = 1
                mask_pred[mask_pred <= 0.5] = 0
                mask_pred[mask_pred > 0.5] = 1

                # center_pred = cv2.erode(center_pred, kernel=np.ones((3,3), np.uint8))
                label_pred = mask_pred*2
                index = (center_pred!=0) & (label_pred!=0)
                label_pred[index] = 1

            #save
            cv2.imwrite(out_dir +'/predict/%s/directions/%s_border.jpg'%(tag, name), res)
            cv2.imwrite(out_dir +'/predict/%s/directions/%s_center.jpg'%(tag, name), res1)
            cv2.imwrite(out_dir +'/predict/%s/predicts/%s_center.jpg'%(tag, name), center_pred*255)
            cv2.imwrite(out_dir +'/predict/%s/predicts/%s_mask.jpg'%(tag, name), mask_pred*255)
            cv2.imwrite(out_dir +'/predict/%s/predicts/%s.png'%(tag, name), label_pred*120)


            # image_show('image', image)
            # image_show('contour_overlay', contour_overlay)
            # cv2.waitKey(1)
            # continue




        #assert(test_num == len(test_loader.sampler))
        log.write('-------------\n')
        log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
        log.write('tag=%s\n'%tag)
        log.write('\n')



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    if 0:
        print('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
        print('** some experiment setting **\n')

        ## net ------------------------------
        initial_checkpoint = \
            RESULTS_DIR + '/unet-se-resnext50-single-1a/checkpoint/00009999_model.pth'

        cfg = Configuration()
        net = Net(cfg).cuda()
        if initial_checkpoint is not None:
            print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
            net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

        print('%s\n\n'%(type(net)))
        print('\n')

        image_name = '0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe'
        label = predict_one_image(image_name, net)
        skimage.io.imshow(label)

    if 1:
        run_predict()

    print('\nsucess!')
