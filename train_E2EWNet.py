import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

from common import *
from data.preprocess import *
from utils.logger   import *
from data.dataset import *
from data.transforms import *
from models.lr_schduler  import *
from models.loss import *

# -------------------------------------------------------------------------------------
from models.w_unet.E2E_WNet import *
Net = E2ENet


WIDTH, HEIGHT = 256,256
# -------------------------------------------------------------------------------------

def train_augment(image, mask, index):
    # illumintaion ------------
    if 1:
        type = random.randint(0,4)
        if type==0:
            image = random_transform(image, u=0.5, func=do_custom_process1, gamma=[0.8,2.5],alpha=[0.7,0.9],beta=[1.0,2.0])

        elif type==1:
            image = random_transform(image, u=0.5, func=do_contrast, alpha=[0.5,2.5])

        elif type==2:
            image = random_transform(image, u=0.5, func=do_gamma, gamma=[1,3])

        elif type==3:
            image = random_transform(image, u=0.5, func=do_clahe, clip=[1,3], grid=[8,16])

        else:
            pass
        #print('illumintaion', image.dtype)

    # filter/noise ------------
    if 1:
        type = random.randint(0,2)
        if type==0:
            image = random_transform(image, u=0.5, func=do_unsharp, size=[9,19], strength=[0.2,0.4], alpha=[4,6])

        elif type==1:
            image = random_transform(image, u=0.5, func=do_speckle_noise, sigma=[0.1,0.5])

        else:
            pass
        #print('filter', image.dtype)

    # geometric ------------
    if 1:
        # type = random.randint(0,2)
        # if type==0:
        #     image, mask = random_transform2(image, mask, u=0.5, func=do_stretch2, scale_x=[1,2], scale_y=[1,1] )
        # if type==1:
        #     image, mask = random_transform2(image, mask, u=0.5, func=do_stretch2, scale_x=[1,1], scale_y=[1,2] )
        
        
        # image, mask = random_transform2(image, mask, u=0.5, func=do_shift_scale_rotate2, dx=[0,0],dy=[0,0], scale=[1/2,2], angle=[-45,45])
        # image, mask = random_transform2(image, mask, u=0.5, func=do_shift_scale_rotate2, dx=[0,0],dy=[0,0], scale=[1/2,2], angle=[-45,45])
        image, mask = random_transform2(image, mask, u=0.5, func=do_elastic_transform2, grid=[8,64], distort=[0,0.5])

        image, mask = random_crop_transform2(image, mask, WIDTH, HEIGHT, u=0.5)
        image, mask = do_flip_transpose2(image, mask, random.randint(0,8))
        #print('geometric', image.dtype)


    #---------------------------------------
    #image, mask = fix_crop_transform2(image, mask, -1,-1,WIDTH, HEIGHT)

    input = torch.from_numpy(image.transpose((2,0,1))).float().div(255)
    dir_from_border = direction_from_border(mask)
    dir_to_center = direction_to_center(mask)
    weight = mask_to_weight(mask)
    center = mask_to_center(mask)
    foreground = mask_to_foreground(mask)
    dir_from_border = torch.from_numpy(dir_from_border).float()      # float tpye ???
    dir_to_center = torch.from_numpy(dir_to_center).float()
    weight = torch.from_numpy(weight).float()
    center = torch.from_numpy(center[np.newaxis,...]).float()
    foreground = torch.from_numpy(foreground[np.newaxis,...]).float()


    return input, dir_from_border, dir_to_center, weight, center, foreground, mask, index


def valid_augment(image, mask, index):

    image, mask = fix_crop_transform2(image, mask, -1,-1, WIDTH, HEIGHT)

    #---------------------------------------
    input = torch.from_numpy(image.transpose((2,0,1))).float().div(255)
    dir_from_border = direction_from_border(mask)
    dir_to_center = direction_to_center(mask)
    weight = mask_to_weight(mask)
    center = mask_to_center(mask)
    foreground = mask_to_foreground(mask)
    dir_from_border = torch.from_numpy(dir_from_border).float()      
    dir_to_center = torch.from_numpy(dir_to_center).float()
    weight = torch.from_numpy(weight).float()      # float tpye ？？？
    center = torch.from_numpy(center[np.newaxis,...]).float()
    foreground = torch.from_numpy(foreground[np.newaxis,...]).float()

    return input, dir_from_border, dir_to_center, weight, center, foreground, mask, index



def train_collate(batch):
    batch_size       = len(batch)
    inputs           = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    dirs_from_border = torch.stack([batch[b][1]for b in range(batch_size)], 0)
    dirs_to_center   = torch.stack([batch[b][2]for b in range(batch_size)], 0)
    weights          = torch.stack([batch[b][3]for b in range(batch_size)], 0)
    centers          = torch.stack([batch[b][4]for b in range(batch_size)], 0)
    foregrounds      = torch.stack([batch[b][5]for b in range(batch_size)], 0)

    masks            =             [batch[b][6]for b in range(batch_size)]
    indices          =             [batch[b][7]for b in range(batch_size)]

    return [inputs, dirs_from_border, dirs_to_center, weights, centers, foregrounds, masks, indices]


################### training ####################
def evaluate( net, test_loader ):

    test_num  = 0
    test_loss = np.zeros(6,np.float32)
    #return test_loss

    for i, (inputs, dirs_from_border, dirs_to_center, weights, ctrs_truth, fgs_truth, masks, indices) in enumerate(test_loader, 0):

        with torch.no_grad():
            inputs       = Variable(inputs).cuda()
            dirs_from_border = Variable(dirs_from_border).cuda()
            dirs_to_center = Variable(dirs_to_center).cuda()
            weights      = Variable(weights).cuda()
            ctrs_truth   = Variable(ctrs_truth).cuda()
            fgs_truth    = Variable(fgs_truth).cuda()

            dirs_from_border_pred, dirs_to_center_pred, ctrs_pred, fgs_pred = net.forward( inputs )
            dir_loss = AngularMSE()( dirs_from_border_pred, dirs_from_border, weights ) + \
                       AngularMSE()( dirs_to_center_pred, dirs_to_center, weights)
            ctr_weights = make_weight( ctrs_truth, 0.2, 0.8 )
            ctr_loss = WeightedBCELoss2d()( ctrs_pred, ctrs_truth, ctr_weights )
            fg_weights = make_weight( fgs_truth, 0.5, 0.5 )
            fg_loss1 = WeightedBCELoss2d()( fgs_pred, fgs_truth, fg_weights )
            fg_loss2 = WeightedSoftDiceLoss()(fgs_pred, fgs_truth, fg_weights)
            fg_loss = fg_loss1 + fg_loss2

        loss = 0.1*dir_loss + 0.6*ctr_loss + 0.3*fg_loss
        batch_size = len(indices)
        test_loss += batch_size*np.array((
                           loss.cpu().data.numpy(),
                           dir_loss.cpu().data.numpy(),
                           ctr_loss.cpu().data.numpy(),
                           fg_loss.cpu().data.numpy(),
                           fg_loss2.cpu().data.numpy(),
                           0,
                         ))
        test_num  += batch_size

    assert(test_num == len(test_loader.sampler))
    test_loss = test_loss/test_num
    return test_loss


#--------------------------------------------------------------
def run_train():

    # RESULTS_DIR is defined in common.py
    out_dir = RESULTS_DIR + '/E2E_WNet-DN-6a'
    initial_checkpoint = \
        None # RESULTS_DIR + '/E2E_WNet-DN-1a/checkpoint/00033400_model.pth'

    pretrain_file = \
        None # RESULTS_DIR + '/WNet-DN-1a/checkpoint/00049999_model.pth'
    ## setup  -----------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH+'/models/w_unet/', out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')



    ## net ----------------------
    log.write('** net setting **\n')
    net = Net(3, 2).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        # cfg = load_pickle_file(out_dir +'/checkpoint/configuration.pkl')

    if pretrain_file is not None:
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        net.load_pretrain( pretrain_file )

    log.write('%s\n\n'%(type(net)))
    log.write('%s\n'%(net.version))
    log.write('\n')


    ## optimiser ----------------------------------
    batch_size  = 8

    num_iters   = 1000  *50
    iter_smooth = 20
    iter_log    = 50
    iter_valid  = 100
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 500))#1*1000

    lr = [0.01, 0.01]
    optimizer = optim.SGD([
                            {'params': filter(lambda p: p.requires_grad, net.dnet.parameters()) },
                            {'params': filter(lambda p: p.requires_grad, net.unet2.parameters())}
                        ], lr=0.01, momentum=0.9, weight_decay=0.0001)

    start_iter = 0
    start_epoch= 0.
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']

        rate = get_learning_rate(optimizer)  #load all except learning rate
        optimizer.load_state_dict(checkpoint['optimizer'])
        adjust_learning_rate(optimizer, rate)
        pass


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = ScienceDataset(
                            DATA_DIR,
                            #'valid1_ids_gray2_38',
                            'train',
                            #'debug1_ids_gray_only_10',
                            #'disk0_ids_dummy_9',
                            #'train1_ids_purple2_80',
                            #'merge1_1',
                            mode='train', transform = train_augment)

    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = train_collate)


    valid_dataset = ScienceDataset(
                            DATA_DIR,
                            'val',
                            #'valid1_ids_gray2_43',
                            #'debug1_ids_gray_only_10',
                            #'disk0_ids_dummy_9',
                            #'valid1_ids_purple2_20',
                            #'merge1_1',
                            mode='train', transform = valid_augment)

    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = train_collate)

    log.write('\tWIDTH, HEIGHT = %d, %d\n'%(WIDTH, HEIGHT))
    log.write('\ttrain_dataset path = %s\n'%(train_dataset.imgs_dir))
    log.write('\tvalid_dataset path = %s\n'%(valid_dataset.imgs_dir))
    log.write('\tlen(train_dataset)  = %d\n'%(len(train_dataset)))
    log.write('\tlen(valid_dataset)  = %d\n'%(len(valid_dataset)))
    log.write('\tlen(train_loader)   = %d\n'%(len(train_loader)))
    log.write('\tlen(valid_loader)   = %d\n'%(len(valid_loader)))
    log.write('\tbatch_size  = %d\n'%(batch_size))
    log.write('\n')


    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' momentum=%f\n'% optimizer.param_groups[0]['momentum'])

    log.write(' images_per_epoch = %d\n\n'%len(train_dataset))
    log.write(' rate                iter   epoch  num   | valid_loss                 | train_loss                 | batch_loss                   |  time          \n')
    log.write('---------------------------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss  = np.zeros(6,np.float32)
    valid_loss  = np.zeros(6,np.float32)
    batch_loss  = np.zeros(6,np.float32)
    rate = [0, 0]

    start = timer()
    j = 0
    i = 0

    while  i < num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6,np.float32)
        sum = 0

        net.set_mode('train')
        optimizer.zero_grad()
        for inputs, dirs_from_border, dirs_to_center, weights, ctrs_truth, fgs_truth, masks, indices in train_loader:
            #if all(len(b)==0 for b in truth_boxes): continue

            batch_size = len(indices)
            i = j + start_iter
            epoch = (i-start_iter)*batch_size/len(train_dataset) + start_epoch
            num_products = epoch*len(train_dataset)

            if i % iter_valid == 0:
                net.set_mode('valid')
                valid_loss = evaluate(net, valid_loader)
                net.set_mode('train')

                print('\r',end='',flush=True)
                log.write('[%0.5f, %0.5f] %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f %0.2f %0.2f| %0.3f   %0.2f %0.2f %0.2f %0.2f| %0.3f   %0.2f %0.2f %0.2f %0.2f| %s\n' % (\
                         rate[0], rate[1], i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], #valid_loss[5],#valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], #train_loss[5],#train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], #batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            #if 1:
            if i in iter_save:
                torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : i,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(i))

            # learning rate schduler -------------
            if epoch > 0 and epoch % 300 == 0:
                lr = [l*0.1 for l in lr]
                adjust_learning_rate(optimizer, lr)

            rate = get_learning_rate(optimizer)


            # one iteration update  -------------
            inputs       = Variable(inputs).cuda()
            dirs_from_border = Variable(dirs_from_border).cuda()
            dirs_to_center = Variable(dirs_to_center).cuda()
            weights      = Variable(weights).cuda()
            ctrs_truth   = Variable(ctrs_truth).cuda()
            fgs_truth    = Variable(fgs_truth).cuda()

            dirs_from_border_pred, dirs_to_center_pred, ctrs_pred, fgs_pred = net.forward( inputs )
            dir_loss = AngularMSE()( dirs_from_border_pred, dirs_from_border, weights ) + \
                       AngularMSE()( dirs_to_center_pred, dirs_to_center, weights)
            ctr_weights = make_weight( ctrs_truth, 0.2, 0.8 )
            ctr_loss = WeightedBCELoss2d()( ctrs_pred, ctrs_truth, ctr_weights )
            fg_weights = make_weight( fgs_truth, 0.5, 0.5 )
            fg_loss1 = WeightedBCELoss2d()( fgs_pred, fgs_truth, fg_weights )
            fg_loss2 = WeightedSoftDiceLoss()(fgs_pred, fgs_truth, fg_weights)
            fg_loss = fg_loss1 + fg_loss2

            loss = 0.1*dir_loss + 0.6*ctr_loss + 0.3*fg_loss

            # accumulated update
            loss.backward()
            #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()


            # print statistics  ------------
            batch_loss = np.array((
                           loss.cpu().data.numpy(),
                           dir_loss.cpu().data.numpy(),
                           ctr_loss.cpu().data.numpy(),
                           fg_loss.cpu().data.numpy(),
                           fg_loss2.cpu().data.numpy(),
                           0,
                         ))

            sum_train_loss += batch_loss
            sum += 1
            if i % iter_smooth == 0:
                train_loss = sum_train_loss/sum
                sum_train_loss = np.zeros(6,np.float32)
                sum = 0


            print('\r[%0.5f, %0.5f] %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f %0.2f %0.2f| %0.3f   %0.2f %0.2f %0.2f %0.2f| %0.3f   %0.2f %0.2f %0.2f %0.2f| %s  %d,%d,%s' % (\
                         rate[0], rate[1], i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], #valid_loss[5],#valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], #train_loss[5],#train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], #batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60) ,i,j, ''), end='',flush=True)#str(inputs.size()))
            j = j + 1


        pass  #-- end of one data loader --
    pass #-- end of all iterations --


    if 1: #save last
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()

    print('\nsucess!')
