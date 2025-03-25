import os
import os.path as osp
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 3407

C.dataset_name = 'Deepcrack'

if C.dataset_name == 'Deepcrack':
    C.num_train_imgs = 300
    C.num_eval_imgs = 237
elif C.dataset_name == 'crack260':
    C.num_train_imgs = 200
    C.num_eval_imgs = 60
elif C.dataset_name == 'CFD':
    C.num_train_imgs = 100
    C.num_eval_imgs = 18
elif C.dataset_name == 'Crack315':
    C.num_train_imgs = 252
    C.num_eval_imgs = 63
C.num_classes = 1
C.class_names =  ['Background', 'Crack']
#
# """Image Config"""
# C.background = 255
C.image_height = 512
C.image_width = 512



C.backbone = 'my_backbone' # Remember change the path below.

C.pretrained_model= '/home/aa.pth'
C.decoder = 'myMambaDecoder'

C.optimizer = 'AdamW'
#
# """Train Config"""
C.lr = 0.001
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 2
C.real_nepochs = 90
C.niters_per_epoch = C.num_train_imgs // C.batch_size

C.warm_up_epoch = 4
#
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

C.checkpoint_start_epoch = 20
C.checkpoint_step = 1

#
if C.dataset_name == 'Deepcrack':
   C.log_dir = osp.abspath('log_final/log_Deepcrack/' + 'log_' + C.dataset_name + '_' + C.backbone + '_0')
elif C.dataset_name == 'crack260':
    C.log_dir = osp.abspath('log_final/log_crack260/' + 'log_' + C.dataset_name + '_' + C.backbone + '_0')
elif C.dataset_name == 'CFD':
    C.log_dir = osp.abspath('log_final/log_CFD/' + 'log_' + C.dataset_name + '_' + C.backbone + '_0')
elif C.dataset_name == 'Crack315':
    C.log_dir = osp.abspath('log_final/log_Crack315/' + 'log_' + C.dataset_name + '_' + C.backbone + '_0')
while(os.path.exists(C.log_dir)):
    C.log_dir=C.log_dir[:-1]+str(int(C.log_dir[-1])+1)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.nepochs = 185
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()