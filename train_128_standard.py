import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2' #'3,2,1,0'

code_dir = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(code_dir + '/com')


from common import *
# from data.load_data   import *
# from data.transform   import *
from train_128_pad_or_resize import run_train


########--------------- model setup --------------############
scale = 'resize'
batch_size = 16
fold_num = 6
model_version = 'resnet34'

depth = 0
use_last = 1


########-------------- training setup --------------############

steps = [1, 2, 3, 4]

schduler1 = None
schduler2 = None
schduler3 = None
schduler4 = None
schduler5 = SnapshotScheduler(0.01, 50, min_lr=0.001)

schduler_list         = [ schduler1, schduler2, schduler3, schduler4, schduler5 ]
lovasz_list           = [ 'focal',   'lovasz',   'lovasz',  'lovasz',  'lovasz' ]
sgd_lr_list           = [   0.01,      0.01,      0.005,     0.001,       0.01  ]

target_loss_list      = [  0.8,      0.84,        0.845,     0.85,       0.85   ]
target_save_loss_list = [  0.78,     0.82,        0.83,      0.83,       0.84   ]
num_epoch_list        = [   30,       200,        100,        60,        200    ]
restart_list          = [   1,         1,          0,         0,          0     ]

sgd_mnt_list = [0.9] * len(sgd_lr_list)
sgd_wd_list = [1e-4] * len(sgd_lr_list)

assert(len(schduler_list) == len(lovasz_list) == len(sgd_lr_list) == len(num_epoch_list) == len(restart_list) )


######## main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    for step in steps:
        ith = int(step) - 1
        lovasz, num_epoch, schduler = lovasz_list[ith], num_epoch_list[ith], schduler_list[ith]
        sgd_lr, sgd_mmt, sgd_wd = sgd_lr_list[ith], sgd_mnt_list[ith], sgd_wd_list[ith]  
        target_loss, restart = target_loss_list[ith], restart_list[ith]
        target_save_loss = target_save_loss_list[ith]

        if str(step) == '1':
            initial_checkpoint = None
        else:
            try:
                saved_df = df[df['model'] != 'not_saved']
                saved_lb = list(saved_df['valid_lb'])
                max_index = np.argsort(saved_lb)[-1]
                initial_checkpoint = saved_df.iloc[max_index]['model']
            except:
                pass

        out_dir = PROJECT_DIR.replace(PROJECT_DIR.split('/')[-1], '')[:-1]

        out_dir += '/results-%s/fold%d-%s128/fold%d-step%s' % (model_version, fold_num, scale, fold_num, step)

        df = run_train(fold_num, out_dir, initial_checkpoint=initial_checkpoint, 
                target_loss=target_loss, target_save_loss=target_save_loss, loss_type=lovasz, restart=restart,
                scale=scale, model_version=model_version, depth_type=depth, 
                num_epoch=num_epoch, batch_size=batch_size, 
                schduler=schduler, sgd_lr=sgd_lr, sgd_mmt=sgd_mmt, sgd_wd=sgd_wd, )


    print('\nsucess!')




