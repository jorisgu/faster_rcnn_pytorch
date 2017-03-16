
# coding: utf-8

# In[1]:
#
# get_ipython().system(u'cd /home/jguerry/workspace/jg_dl/faster_rcnn_pytorch')
#
#
# # In[2]:
#
pytorchpath = '/data02/jguerry/jg_pyt/'
# import sys
# if not pytorchpath in sys.path:
#     sys.path.append(pytorchpath)




import os
import torch
import numpy as np
from datetime import datetime

from faster_rcnn import network
from faster_rcnn.faster_rcnn_y import FasterRCNN_y
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer_fuse import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

try:
    from termcolor import cprint
except ImportError:
    cprint = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



# hyper-parameters
# ------------
imdb_name_0 = 'inout_train_Images'
imdb_name_1 = 'inout_train_Depth'
cfg_file = pytorchpath+'experiments/cfgs/faster_rcnn_end2end_inout.yml'
pretrained_model = pytorchpath+'data/pretrained_model/VGG_imagenet.npy'
output_dir = pytorchpath+'models/inout_y/'

start_step = 0
end_step = 100000
lr_decay_steps = {60000, 80000}
lr_decay = 1./10

rand_seed = 1024
_DEBUG = True
use_tensorboard = True
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
imdb_0 = get_imdb(imdb_name_0)
imdb_1 = get_imdb(imdb_name_1)
rdl_roidb.prepare_roidb(imdb_0)
rdl_roidb.prepare_roidb(imdb_1)
roidb_0 = imdb_0.roidb
roidb_1 = imdb_1.roidb
data_layer = RoIDataLayer(roidb_0, roidb_1, imdb_0.num_classes)


# In[5]:

# load net
net = FasterRCNN_y(classes=imdb_0.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)
network.load_pretrained_npy_y(net, pretrained_model)
# model_file = '/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5'
# model_file = 'models/saved_model3/faster_rcnn_60000.h5'
# network.load_net(model_file, net)
# exp_name = 'vgg16_02-19_13-24'
# start_step = 60001
# lr /= 10.
# network.weights_normal_init([net.bbox_fc, net.score_fc, net.fc6, net.fc7], dev=0.01)

net.cuda()
net.train()

# params = list(net.parameters())


# In[6]:

frozen_params = map(id, net.rpn_0.features.conv1.parameters())+map(id, net.rpn_0.features.conv2.parameters())+map(id, net.rpn_1.features.conv1.parameters())+map(id, net.rpn_1.features.conv2.parameters())

base_params = filter(lambda p: id(p) not in frozen_params, net.parameters())


# In[7]:

# optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.SGD(base_params, lr=lr, momentum=momentum, weight_decay=weight_decay)


# In[15]:

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
for step in range(start_step, end_step+1):

    # get one batch
    blobs = data_layer.forward()
    im_data_0 = blobs['data']
    im_data_1 = blobs['data_']
    im_info = blobs['im_info']
    gt_boxes = blobs['gt_boxes']
    gt_ishard = blobs['gt_ishard']
    dontcare_areas = blobs['dontcare_areas']

    # forward
    net(im_data_0, im_data_1, im_info, gt_boxes, gt_ishard, dontcare_areas)
    loss = net.loss + net.rpn_0.loss + net.rpn_1.loss

    if _DEBUG:
        tp += float(net.tp)
        tf += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt

    train_loss += loss.data[0]
    step_cnt += 1

    # backward
    optimizer.zero_grad()
    loss.backward()
    network.clip_gradient(net, 10.)
    optimizer.step()

    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps)
        log_print(log_text, color='green', attrs=['bold'])

        if _DEBUG:
            log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
            log_print('\trpn0_cls: %.4f, rpn0_box: %.4f,rpn1_cls: %.4f, rpn1_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.rpn_0.cross_entropy.data.cpu().numpy()[0], net.rpn_0.loss_box.data.cpu().numpy()[0],
                net.rpn_1.cross_entropy.data.cpu().numpy()[0], net.rpn_1.loss_box.data.cpu().numpy()[0],
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )
        re_cnt = True


    if (step % 10000 == 0) and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))
    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(base_params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False


# In[ ]:
