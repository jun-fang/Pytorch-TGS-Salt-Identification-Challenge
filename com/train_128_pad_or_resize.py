import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
 
from common import *

SIZE = 101
RESIZE  = 128

PAD  = 14
Y0, Y1, X0, X1 = PAD,PAD+SIZE,PAD,PAD+SIZE,


## preload resnet34
def load_old_pretrain_file(net, pretrain_file, skip=[]):

    pretrain_state_dict = torch.load(pretrain_file)
    state_dict = net.state_dict()
    keys = list(state_dict.keys())

    for key in keys:
        if any(s in key for s in skip):
            continue

        if  'encoder1.' in key:
        # if 'conv1.' in key:
            key0 = key.replace('encoder1.0','conv1').replace('encoder1.1','bn1')
            state_dict[key] = pretrain_state_dict[key0]
            continue

        if 'encoder2.' in key:
            key0 = key.replace('encoder2.1.','layer1.')
            state_dict[key] = pretrain_state_dict[key0]
            continue

        if 'encoder3.' in key:
            key0 = key.replace('encoder3.','layer2.')
            state_dict[key] = pretrain_state_dict[key0]
            continue

        if 'encoder4.' in key:
            key0 = key.replace('encoder4.','layer3.')
            state_dict[key] = pretrain_state_dict[key0]
            continue

        if 'encoder5.' in key:
            key0 = key.replace('encoder5.','layer4.')
            state_dict[key] = pretrain_state_dict[key0]
            continue
    
    net.load_state_dict(state_dict)
    return net


def valid_augment(image, mask, index, scale='pad'):
    cache = Struct(image = image.copy(), mask = mask.copy())
    if scale == 'resize':
        image, mask = do_resize2(image, mask, RESIZE, RESIZE)
    elif scale == 'pad':
        image, mask = do_center_pad2(image, mask, PAD)
        image, mask = image[:-1, :-1], mask[:-1, :-1]
    return image,mask,index,cache


def train_augment(image, mask, index, scale='pad'):
    cache = Struct(image = image.copy(), mask = mask.copy())

    if np.random.rand() < 0.5:
         image, mask = do_horizontal_flip2(image, mask)
         pass

    if np.random.rand() < 0.5:
        c = np.random.choice(4)
        if c==0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2) #0.125

        if c==1:
            image, mask = do_horizontal_shear2( image, mask, dx=np.random.uniform(-0.07,0.07) )
            pass

        if c==2:
            image, mask = do_shift_scale_rotate2( image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0,15))  #10

        if c==3:
            image, mask = do_elastic_transform2(image, mask, grid=10, distort=np.random.uniform(0,0.15))#0.10
            pass


    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c==0:
            image = do_brightness_shift(image,np.random.uniform(-0.1,+0.1))
        if c==1:
            image = do_brightness_multiply(image,np.random.uniform(1-0.08,1+0.08))
        if c==2:
            image = do_gamma(image,np.random.uniform(1-0.08,1+0.08))
        # if c==1:
        #     image = do_invert_intensity(image)

    if scale == 'resize':
        image, mask = do_resize2(image, mask, RESIZE, RESIZE)
    if scale == 'pad':
        image, mask = do_center_pad2(image, mask, PAD)
        image, mask = image[:-1, :-1], mask[:-1, :-1]
    return image,mask,index,cache


### training ##############################################################

def do_valid( net, valid_loader, scale='pad'):

    valid_num  = 0
    losses = np.zeros(4,np.float32)

    predicts = []
    truths   = []
    corrects = []

    for input, truth, index, cache in valid_loader:
        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit, logit_pixel, logit_image = data_parallel(net,input) #net(input)
            prob  = F.sigmoid(logit)

            loss_seg, loss_pixel, loss_image = net.criterion(logit, logit_pixel, logit_image, truth)
            # loss  = loss_seg + 0.1 * loss_pixel + 0.05 * loss_image
            dice  = net.metric(logit, truth)

        batch_size = len(index)
        losses += batch_size*np.array(( loss_seg.item(), loss_pixel.item(), loss_image.item(), dice.item()))
        valid_num += batch_size
 
        
        if scale == 'resize':
            prob_shape = prob.data.cpu().numpy().shape
            prob = prob.data.cpu().numpy()
            prob = np.array([do_resize(prob[i, 0, :, :], SIZE, SIZE) for i in range(batch_size)])
            prob = np.array(prob.reshape(batch_size, 1, SIZE, SIZE))
            predicts.append(prob)

            corrects.append(logit_image.data.cpu().numpy())
            
            for c in cache:
                 truths.append(c.mask)
        elif scale == 'pad':
            prob  = prob [:,:,Y0:Y1, X0:X1]
            predicts.append(prob.data.cpu().numpy())

            truth = truth[:,:,Y0:Y1, X0:X1]
            truths.append(truth.data.cpu().numpy())

            corrects.append(logit_image.data.cpu().numpy())

    assert(valid_num == len(valid_loader.sampler))
    losses  = losses / valid_num

    #--------------------------------------------------------
    predicts = np.concatenate(predicts).squeeze()
    corrects  = np.concatenate(corrects)
    if scale == 'resize':
        truths   = np.array(truths)
    elif scale == 'pad':
        truths   = np.concatenate(truths).squeeze()

    c = corrects < 0.5
    empty_index = truths.reshape(valid_num, -1).sum(1) < 1
    non_empty_index = ~ empty_index
    empty_num = empty_index.sum()
    non_empty_num = non_empty_index.sum()
    
    p = (predicts.reshape(valid_num, -1) > 0.5).sum(1) < 1
    t = empty_index

    empty_tp = ((p == 1) * ( t == 1)).sum() / valid_num
    empty_fp = ((p == 0) * ( t == 1)).sum() / valid_num

    non_empty_tp = ((p == 0) * ( t == 0) * ( c == 0)).sum() / valid_num
    non_empty_fp = ((p == 0) * ( t == 0) * ( c == 1)).sum() / valid_num
    non_empty_fn = ((p == 1) * ( t == 0)).sum() / valid_num 

    precision, result, threshold  = do_kaggle_metric(predicts[empty_index], truths[empty_index])
    precision_empty = precision.mean()
    correct_empty = corrects[empty_index].mean()

    precision, result, threshold  = do_kaggle_metric(predicts[non_empty_index], truths[non_empty_index])
    precision_non_empty = precision.mean()
    correct_non_empty = corrects[non_empty_index].mean()

    precision = (empty_num * precision_empty + non_empty_num * precision_non_empty) / valid_num

    valid_loss = np.array([
        losses[0], losses[3], precision, # all images
        losses[2], precision_empty, empty_tp, empty_fp,  # empty
        losses[1], precision_non_empty, non_empty_tp, non_empty_fp, non_empty_fn, # non-empty
        ])

    return valid_loss


### training ##############################################################

def run_train(fold_num, out_dir, initial_checkpoint=None, target_loss=0.845, target_save_loss=None, scale='pad', model_version='resnet34', loss_type='focal', depth_type=0,  
    num_epoch=25, batch_size=16, schduler=None, sgd_lr=0.05, sgd_mmt=0.9, sgd_wd=1e-4, save_training_epoch_batch=5, restart=False):

    if target_save_loss is None:
        target_save_loss = target_loss - 0.01

    ########-------------------------------------------############
    ## import model
    if model_version == 'resnet34':
        from model.model_resnet_dss import UNetResNet34 as Net
    elif model_version == 'se_resnext50':
        from model.model_senet_dss import UNetSEResNext50 as Net
    elif model_version == 'senet154':
        from model.model_senet_dss import UNetSENext154 as Net
    
    ## pre-train model
    if initial_checkpoint is None:
        if model_version.startswith('resnet34'):
            pretrain_file = DATA_DIR + '/model/resnet34-333f7ec4.pth'
        elif model_version.startswith('se_resnext50'):
            pretrain_file = DATA_DIR + '/model/se_resnext50_32x4d-a260b3a4.pth'
        elif model_version.startswith('senet154'):
            pretrain_file = DATA_DIR + '/model/senet154-c7b49a05.pth'
    else:
        pretrain_file = None 
        assert(('fold%s' % fold_num) in initial_checkpoint or ('fold-%s' % fold_num) in initial_checkpoint)  

    ## record training details 
    training_record_folder = out_dir + os.sep + 'train'
    training_record_paths = sorted(gb(training_record_folder + '/*.xlsx'))
    training_counter = len(training_record_paths) + 1
    training_record = out_dir + '/train/training_record_%s.xlsx' % (str(100 + training_counter)[1:])

    ## setup  -----------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/train', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_DIR, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    # log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_DIR) 
    log.write('\tDATA_PATH    = %s\n' % DATA_DIR)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\tfold_num     = %s\n' % fold_num)
    log.write('\ttarget_loss  = %s\n' % target_loss)
    log.write('\tt_save_loss  = %s\n' % target_save_loss)
    log.write('\tloss_type    = %s\n' % loss_type)
    log.write('\tdepth_type   = %s\n' % depth_type)
    log.write('\tnum_epoch    = %s\n' % num_epoch)
    log.write('\tmomentum     = %s\n' % sgd_mmt)
    log.write('\tweight_decay = %s\n' % sgd_wd)
    log.write('\ttrain_record = %s\n' % training_record)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... \n')
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = batch_size

    train_dataset = TsgDataset('list_train%d_3600' % (fold_num), train_augment, 'train', scale=scale)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler     = RandomSampler(train_dataset),
                        #sampler     = ConstantSampler(train_dataset,[31]*batch_size*100),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    valid_dataset = TsgDataset('list_valid%d_400' % (fold_num), valid_augment, 'train', scale=scale)
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = RandomSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset.split = %s\n'%(train_dataset.split))
    log.write('valid_dataset.split = %s\n'%(valid_dataset.split))
    log.write('\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(loss_type=loss_type, depth_type=depth_type).cuda()


    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    if pretrain_file is not None:
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        if model_version.startswith('resnet34'):
            net = load_old_pretrain_file(net, pretrain_file, skip=['num_batches_tracked', 'scale'])
        elif model_version.startswith('se_resnext50') or model_version.startswith('senet154'):
            net.load_pretrain(pretrain_file)


    ## optimizer ----------------------------------
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=sgd_lr, momentum=sgd_mmt, weight_decay=sgd_wd)
    if isinstance(schduler, list) and schduler[0] == 'ReduceLROnPlateau':
        lr_factor = schduler[1]
        lr_patience = schduler[2]
        schduler = ReduceLROnPlateau(optimizer, mode='max', factor=lr_factor, patience=lr_patience, verbose=False, min_lr=1e-5)
        log.write('\nUsing ReduceLROnPlateau: factor = %.2f, patience = %d ...' % (lr_factor, lr_patience))

    log.write('\nnetwork: %s\n'%(type(net)))
    log.write('schduler: %s\n'%(type(schduler)))
    log.write('\n')

    ## record training ----------------------------------
    rate_list, iter_list, epoch_list = [], [], []
    valid_loss_list, valid_acc_list, valid_lb_list = [], [], []
    train_loss_list, trian_acc_list, batch_loss_list, batch_acc_list = [], [], [], []
    current_time, running_time, model_name_list, lovasz_list, fold_list = [], [], [], [], []

    start_iter = 0
    start_epoch= 0
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('_model','_optimizer'))
        if not restart:
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']

        rate = get_learning_rate(optimizer)  #load all except learning rate
        #optimizer.load_state_dict(checkpoint['optimizer'])
        adjust_learning_rate(optimizer, rate)
        pass
    
    len_train_dataset = len(train_dataset)
    num_iter_per_epoch = int(len_train_dataset * 1.0 / batch_size)
    num_iters   = min(int(start_iter + num_epoch * num_iter_per_epoch), 300  *1000)
    iter_smooth = 50
    iter_valid  = 100
    iter_save   = [0, num_iters-1]\
                   + list(range(0, 300  *1000, 500))


    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' samples_per_epoch = %d\n\n'%len(train_dataset))
    log.write('                         | ------------------- VILID ----------------------------------------------------------------    | ---- TRAIN ----| --- BATCH ---  |          \n')
    log.write('                         | ---------- all -------| ------------ empty -----------| --------------- non-empty ----------- |          \n')
    log.write(' rate     iter   epoch   | loss    acc     lb    |  loss     lb      tp      fp  |  loss     lb       tp     fp     fn   | train_loss     | batch_loss     |  time          \n')
    log.write('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss  = np.zeros(6,np.float32)
    valid_loss  = np.zeros(6,np.float32)
    batch_loss  = np.zeros(6,np.float32)
    rate = 0
    iter = 0
    i    = 0

    start = timer()
    while  iter<num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6,np.float32)
        sum = 0

        optimizer.zero_grad()
        for input, truth, index, cache in train_loader:

            len_train_dataset = len(train_dataset)
            batch_size = len(index)
            iter = i + start_iter
            epoch = (iter-start_iter)*batch_size/len_train_dataset + start_epoch
            num_samples = epoch*len_train_dataset

            if iter % iter_valid==0:
                net.set_mode('valid')
                valid_loss = do_valid(net, valid_loader, scale=scale)
                local_lb = valid_loss[2]
                # valid_loss = np.array([
                #     losses[0], losses[3], precision, # all images
                #     losses[2], precision_empty, empty_tp, empty_fp,  # empty
                #     losses[1], precision_non_empty, non_empty_tp, non_empty_fp, non_empty_fn, # non-empty
                #     ])

                net.set_mode('train')

                model_name = 'not_saved'

                ## save for good ones
                if local_lb >= target_loss:
                    log.write('\n')
                    log.write('\n    save good model at iter: %5.1fk,    local lb: %.6f\n\n' % (iter/1000, local_lb))
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model_%.6f.pth'%(iter, local_lb))
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'iter'     : iter,
                        'epoch'    : epoch,
                    }, out_dir +'/checkpoint/%08d_optimizer_%.6f.pth'%(iter, local_lb))

                    model_name = out_dir +'/checkpoint/%08d_model_%.6f.pth'%(iter, local_lb)
                    pass

                elif iter in iter_save and local_lb >= target_save_loss:
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'iter'     : iter,
                        'epoch'    : epoch,
                    }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))

                    model_name = out_dir +'/checkpoint/%08d_model.pth'%(iter)
                    pass
                    
                print('\r',end='',flush=True)
                log.write('%0.5f  %5.1f  %6.1f  |  %0.3f  %0.3f  (%0.3f) |  %0.3f  (%0.3f)  %0.3f  %0.3f |  %0.3f  (%0.3f)  %0.3f  %0.3f  %0.3f  |  %0.3f  %0.3f  |  %0.3f  %0.3f  | %s \n' % (\
                         rate, iter/1000, epoch,
                         valid_loss[0], valid_loss[1], local_lb,
                         valid_loss[3], valid_loss[4], valid_loss[5], valid_loss[6],
                         valid_loss[7], valid_loss[8], valid_loss[9], valid_loss[10], valid_loss[11],
                         train_loss[0], train_loss[1],
                         batch_loss[0], batch_loss[1],
                         time_to_str((timer() - start))))
                time.sleep(0.01)

                rate_list.append(rate)
                iter_list.append(iter)
                epoch_list.append(epoch)
                
                valid_loss_list.append(valid_loss[0])
                valid_acc_list.append(valid_loss[1])
                valid_lb_list.append(local_lb)

                train_loss_list.append(train_loss[0])
                trian_acc_list.append(train_loss[1])
                batch_loss_list.append(batch_loss[0])
                batch_acc_list.append(batch_loss[1])

                current_time.append(strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                running_time.append(time_to_str((timer() - start)))
                model_name_list.append(model_name)
                lovasz_list.append(loss_type)
                fold_list.append(fold_num)

            if int(epoch - start_epoch + 1) % save_training_epoch_batch == 0:
                temp_df = pd.DataFrame({
                    'rate': rate_list, 'iter': iter_list, 'epoch': epoch_list,
                    'valid_loss': valid_loss_list, 'valid_acc': valid_acc_list, 'valid_lb': valid_lb_list,
                    'train_loss': train_loss_list, 'train_acc': trian_acc_list,
                    'batch_loss': batch_loss_list, 'batch_acc': batch_loss_list,
                    'run_time': running_time, 'current_time': current_time,
                    'lovasz': lovasz_list, 'fold': fold_list, 'model': model_name_list
                    })
                temp_df = temp_df[['rate', 'iter', 'epoch', 'valid_loss', 'valid_acc', 'valid_lb',
                    'train_loss', 'train_acc', 'batch_loss', 'batch_acc',
                    'run_time', 'current_time', 'lovasz', 'fold', 'model']]
                temp_df.to_excel(training_record, index=False)   

            #learning rate schduler -------------
            if schduler is not None:
                if str(schduler).startswith('Snapshot'):
                    lr = schduler.get_rate(epoch - start_epoch)
                    if lr<0 : break
                    adjust_learning_rate(optimizer, lr)
                elif 'ReduceLROnPlateau' in str(schduler):
                    if iter % int(len_train_dataset / batch_size) == 0: 
                        schduler.step(valid_loss[2])
                else:
                    lr = schduler.get_rate(iter)
                    if lr<0 : break
                    adjust_learning_rate(optimizer, lr)
                
            rate = get_learning_rate(optimizer)
            

            # one iteration update  -------------
            net.set_mode('train')
            input = input.cuda()
            truth = truth.cuda()
            logit, logit_pixel, logit_image = data_parallel(net,input) #net(input)
            loss_seg, loss_pixel, loss_image = net.criterion(logit, logit_pixel, logit_image, truth)
            loss  = loss_seg + 0.1 * loss_pixel + 0.05 * loss_image
            dice  = net.metric(logit, truth)
            
            ## original SGD
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # print statistics flush ------------
            batch_loss = np.array((
                           loss.item(),
                           dice.item(),
                           0, 0, 0, 0,
                         ))
            sum_train_loss += batch_loss
            sum += 1
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/sum
                sum_train_loss = np.zeros(6,np.float32)
                sum = 0


            print('\r%0.5f  %5.1f  %6.1f  |  %0.3f  %0.3f  (%0.3f) |------------------------------------------------------------------->>> |  %0.3f  %0.3f  |  %0.3f  %0.3f  | %s ' % (\
                         rate, iter/1000, epoch,
                         valid_loss[0], valid_loss[1], valid_loss[2],
                         train_loss[0], train_loss[1],
                         batch_loss[0], batch_loss[1],
                         time_to_str((timer() - start))), end='',flush=True)
            i=i+1

        pass  #-- end of one data loader --
    

    df = pd.DataFrame({
        'rate': rate_list, 'iter': iter_list, 'epoch': epoch_list,
        'valid_loss': valid_loss_list, 'valid_acc': valid_acc_list, 'valid_lb': valid_lb_list,
        'train_loss': train_loss_list, 'train_acc': trian_acc_list,
        'batch_loss': batch_loss_list, 'batch_acc': batch_loss_list,
        'run_time': running_time, 'current_time': current_time,
        'lovasz': lovasz_list, 'fold': fold_list, 'model': model_name_list
        })
    df = df[['rate', 'iter', 'epoch', 'valid_loss', 'valid_acc', 'valid_lb',
        'train_loss', 'train_acc', 'batch_loss', 'batch_acc',
        'run_time', 'current_time', 'lovasz', 'fold', 'model']]
    df.to_excel(training_record, index=False) 

    pass #-- end of all iterations --

    if 0: #save last
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')

    return df

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()

    print('\nsucess!')
