import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
SIZE = 101
RESIZE  = 128

PAD  = 14
Y0, Y1, X0, X1 = PAD,PAD+SIZE,PAD,PAD+SIZE,

from common import *


### submitting   ##############################################################

#augment == 'flip'
def test_augment_flip(image, mask, index, scale='pad'):
    cache = Struct(image = image.copy(), mask = mask.copy())

    if mask==[]:
        image = do_horizontal_flip(image)
        if scale == 'pad':
            image = do_center_pad(image, PAD)
            image = image[:-1, :-1]
        elif scale == 'resize':
            image = do_resize(image, RESIZE, RESIZE)
    else:
        image, mask = do_horizontal_flip2(image, mask)
        if scale == 'pad':
            image, mask = do_center_pad2(image, mask, PAD)
            image, mask = image[:-1, :-1], mask[:-1, :-1]
        elif scale == 'resize':
            image, mask = do_resize2(image, mask,  RESIZE, RESIZE)
    return image, mask, index, cache


def test_unaugment_flip(prob):
    # dy0, dy1, dx0, dx1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)
    # prob = prob[:,dy0:dy0+IMAGE_HEIGHT, dx0:dx0+IMAGE_WIDTH]
    prob = prob[:,:,::-1]
    return prob

#---------------------
#augment == 'null' :
def test_augment_null(image, mask, index, scale='pad'):
    cache = Struct(image = image.copy(), mask = mask.copy())

    if mask==[]:
        if scale == 'pad':
            image = do_center_pad(image, PAD)
            image = image[:-1, :-1]
        elif scale == 'resize':
            image = do_resize(image, RESIZE, RESIZE)
    else:
        if scale == 'pad':
            image, mask = do_center_pad2(image, mask, PAD)
            image, mask = image[:-1, :-1], mask[:-1, :-1]
        elif scale == 'resize':
            image, mask = do_resize2(image, mask,  RESIZE, RESIZE)
    return image, mask, index, cache


def test_unaugment_null(prob):
    # dy0, dy1, dx0, dx1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)
    # prob = prob[:,dy0:dy0+IMAGE_HEIGHT, dx0:dx0+IMAGE_WIDTH]
    return prob



##############################################################################################

def run_predict(out_dir, initial_checkpoint, split, mode, augment, save_iter=False, batch_size=32, 
        model_version='resnet34', loss_type='focal', depth_type=0, scale='pad'):

    if augment == 'null':
        test_augment   = test_augment_null
        test_unaugment = test_unaugment_null
    if augment == 'flip':
        test_augment   = test_augment_flip
        test_unaugment = test_unaugment_flip
    #....................................................

    ## import model
    if model_version == 'resnet34':
        from model.model_resnet_dss import UNetResNet34 as Net
    elif model_version == 'se_resnext50':
        from model.model_senet_dss import UNetSEResNext50 as Net
    elif model_version == 'senet154':
        from model.model_senet_dss import UNetSENext154 as Net


    ## setup  -----------------
    os.makedirs(out_dir +'/test/' + split, exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.test.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [run_predict %s] %s\n\n' % (strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '-' * 64))
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\tsplit        = %s\n' % split)
    log.write('\tmode         = %s\n' % mode)
    log.write('\taugment      = %s\n' % augment)
    log.write('\tscale        = %s\n' % scale)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    # batch_size = 32

    test_dataset = TsgDataset(split, test_augment, mode, scale=scale)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    assert(len(test_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(loss_type=loss_type, depth_type=depth_type).cuda()

    log.write('\nnetwork: %s\n'%(type(net)))


    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n'%(type(net)))
    log.write('\n')

    ####### start here ##########################
    all_prob = []
    all_num  = 0
    all_loss = np.zeros(2,np.float32)

    net.set_mode('test')
    for input, truth, index, cache in test_loader:
        print('\r',all_num, end='', flush=True)
        batch_size = len(index)
        all_num += batch_size

        input = input.cuda()
        with torch.no_grad():
            logit, logit_pixel, logit_image = data_parallel(net,input) #net(input)
            prob  = F.sigmoid(logit)

            if scale == 'resize':
                prob_shape = prob.data.cpu().numpy().shape
                prob = prob.data.cpu().numpy()
                prob = np.array([do_resize(prob[i, 0, :, :], SIZE, SIZE) for i in range(batch_size)])
                prob = np.array(prob.reshape(batch_size, 1, SIZE, SIZE))
            
            elif scale == 'pad':
                prob  = prob [:,:,Y0:Y1, X0:X1]
                prob = prob.data.cpu().numpy()

            if mode == 'valid': ##for debug
                truth = truth.cuda()
                loss_seg, loss_pixel, loss_image = net.criterion(logit, logit_pixel, logit_image, truth)
                loss  = loss_seg #+ 0.1 * loss_pixel + 0.05 * loss_image
                dice  = net.metric(logit, truth)
                all_loss += batch_size*np.array(( loss.item(), dice.item(),))

        prob = prob.squeeze()
        prob = test_unaugment(prob)
        all_prob.append(prob)

    print('\r',all_num, end='\n', flush=True)
    all_prob = np.concatenate(all_prob)
    all_prob = (all_prob*65535).astype(np.uint16)
    
    if save_iter:
        iter_num = initial_checkpoint.split('/')[-1].split('_')[0]
        save_iter = out_dir +'/test/%s/%s-%s-%s.prob.uint16.npy'%(split, iter_num, split, augment)
        np.save(save_iter, all_prob)
        # np.savez_compressed(save_iter + '.npz', all_prob)
    save_name_prob = out_dir +'/test/%s-%s.prob.uint16.npy'%(split,augment)
    np.save(save_name_prob ,all_prob)
    print(all_prob.shape)

    print('')
    assert(all_num == len(test_loader.sampler))
    all_loss  = all_loss/all_num
    print(all_loss)
    log.write('\nlovash loss: %.6f\n' % all_loss[0])
    log.write('\naccuracy: %.6f\n' % all_loss[1])
    log.write('\n')

    return all_loss


def run_submit(out_dir, initial_checkpoint, split, mode, augment, save_name=None, augmentation_paths=None, save_iter=False, scale='pad'):

    if augment in ['null','flip']:

        augmentation = [
            1, out_dir + '/test/%s-%s.prob.uint16.npy'%(split,augment),
        ]
        csv_file = out_dir + '/test/%s-%s.csv.gz'%(split,augment)


    if augment == 'aug2':
        augmentation = [
            1, out_dir + '/test/%s-%s.prob.uint16.npy'%(split,'null'),
            1, out_dir + '/test/%s-%s.prob.uint16.npy'%(split,'flip'),
        ]
        # print(augmentation)
        csv_file = out_dir + '/test/%s-%s.csv.gz'%(split,augment)

    os.makedirs(out_dir +'/test/' + split, exist_ok=True)

    if augmentation_paths and len(augmentation_paths) > 0:
        augmentation = [] 
        for aug in augmentation_paths:
            augmentation.append(1.0)
            augmentation.append(aug)
        print(augmentation)

    #save
    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [run_submit %s] %s\n\n' % (strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '-' * 64))
    log.write('\tsplit        = %s\n' % split)
    log.write('\tmode         = %s\n' % mode)
    log.write('\taugment      = %s\n' % augment)
    log.write('\n')
    log.write('\taugmentation = %s\n' % augmentation)
    log.write('\n')


    augmentation = np.array(augmentation, dtype=object).reshape(-1,2)
    num_augments = len(augmentation)
    w, augment_file = augmentation[0]
    # all_prob = w*np.load(augment_file).astype(np.float32)/255
    all_prob = w*np.load(augment_file).astype(np.float32)
    if augment_file.endswith('prob.uint8.npy'):
        all_prob = all_prob/255.0
    elif augment_file.endswith('prob.uint16.npy'):
        all_prob = all_prob/65535.0
    all_w = w
    for i in range(1, num_augments):
        w, augment_file = augmentation[i]
        # prob = w*np.load(augment_file).astype(np.float32)/255
        prob = w*np.load(augment_file).astype(np.float32)
        if augment_file.endswith('prob.uint8.npy'):
            prob = prob/255.0
        elif augment_file.endswith('prob.uint16.npy'):
            prob = prob/65535.0
        all_prob += prob
        all_w += w
    all_prob /= all_w

    if save_iter and augment == 'aug2':
        os.makedirs(out_dir +'/sub/%s' % split, exist_ok=True)
        iter_num = initial_checkpoint.split('/')[-1].split('_')[0]
        save_iter = out_dir +'/sub/%s/%s-%s-%s-%s.prob.uint16.npy'%(split, scale, iter_num, split, augment)
        np.save(save_iter, (all_prob*65535).astype(np.uint16))

    if augmentation_paths and len(augmentation_paths) > 0:
        os.makedirs(out_dir +'/sub/ensemble', exist_ok=True)
        save_iter = out_dir +'/sub/ensemble/%s-%s.prob.uint16.npy'%(save_name, split)
        np.save(save_iter, (all_prob*65535).astype(np.uint16))

    all_prob = all_prob>0.5
    print(all_prob.shape)

    #----------------------------
    split_file = '../data/split/' + split
    lines = read_list_from_file(split_file)

    id = []
    rle_mask = []
    for n, line in enumerate(lines):
        folder, name = line.split('/')
        id.append(name)

        # if (all_prob[n].sum()<=0):
        #     encoding=''
        if (all_prob[n].sum()<=0 or all_prob[n].mean() == 1):
            encoding=''
        else:
            encoding = do_length_encode(all_prob[n])
            # if len(encoding.split(' ')) < 10:
            #     encoding = ''
        assert(encoding!=[])

        rle_mask.append(encoding)

    df = pd.DataFrame({ 'id' : id , 'rle_mask' : rle_mask}).astype(str)
    df.to_csv(csv_file, index=False, columns=['id', 'rle_mask'], compression='gzip')

    if save_name:
        os.makedirs(out_dir +'/sub', exist_ok=True)
        csv_file = out_dir + '/sub/%s_%s.csv' % (save_name, augment)
        df.to_csv(csv_file, index=False, columns=['id', 'rle_mask'])


############################################################################################
def run_local_leaderboard(out_dir, initial_checkpoint, split, mode, augment):

    #-----------------------------------------------------------------------
    submit_file = out_dir + '/test/%s-%s.csv.gz'%(split,augment)
    # dump_dir = out_dir + '/test/%s-%s-dump'%(split,augment)
    # os.makedirs(dump_dir, exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')

    split_file = '../data/split/' + split
    lines = read_list_from_file(split_file)
    ids = [line.split('/')[-1] for line in lines]
    sorted(ids)

    df_submit = pd.read_csv(submit_file).set_index('id')
    df_submit = df_submit.fillna('')

    df_truth  = pd.read_csv('../data/train.csv').set_index('id')
    df_truth  = df_truth.loc[ids]
    df_truth  = df_truth.fillna('')

    N = len(df_truth)
    predict = np.zeros((N,101,101),np.bool)
    truth   = np.zeros((N,101,101),np.bool)

    for n in  range(N):
        id = ids[n]
        p  = df_submit.loc[id].rle_mask
        t  = df_truth.loc[id].rle_mask
        p  = do_length_decode(p, H=101, W=101, fill_value=1).astype(np.bool)
        t  = do_length_decode(t, H=101, W=101, fill_value=1).astype(np.bool)

        predict[n]=p
        truth[n]=t

    ##--------------
    precision, result, threshold = do_kaggle_metric(predict,truth, threshold=0.5)
    precision_mean = precision.mean()

    tp, fp, fn, tn_empty, fp_empty = result.transpose(1,2,0).sum(2)
    all = tp + fp + fn + tn_empty + fp_empty
    p   = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)


    log.write('\n')
    log.write('** %s ** \n'%augment)
    log.write('      |        |                                      |           empty          |         \n')
    log.write('th    |  prec  |      tp          fp          fn      |      tn          fp      |         \n')
    log.write('-------------------------------------------------------------------------------------------\n')
    for i, t in enumerate(threshold):
        log.write('%0.2f  |  %0.2f  |  %3d / %0.2f  %3d / %0.2f  %3d / %0.2f  |  %3d / %0.2f  %3d / %0.2f  | %5d\n'%(
            t, p[i],
            tp[i], tp[i]/all[i],
            fp[i], fp[i]/all[i],
            fn[i], fn[i]/all[i],
            tn_empty[i], tn_empty[i]/all[i],
            fp_empty[i], fp_empty[i]/all[i],
            all[i])
        )

    log.write('\n')
    log.write('num images :    %d\n'%N)
    log.write('LB score   : %0.5f\n'%(precision_mean))

    #--------------------------------------
    predict = predict.reshape(N,-1)
    truth   = truth.reshape(N,-1)
    p = predict>0.5
    t = truth>0.5
    intersection = t & p
    union        = t | p
    #iou = intersection.sum(1)/(union.sum(1)+EPS)
    log.write('iou        : %0.5f\n'%(intersection.sum()/(union.sum()+EPS)))

    return precision_mean, intersection.sum()/(union.sum()+EPS)
    