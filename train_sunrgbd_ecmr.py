print "Starting script..."
import operator
import os
import torch
import numpy as np
from datetime import datetime

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
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

print "Loading configuration..."

pytorchpath = os.environ['PWD']+'/'

# hyper-parameters
# ------------
imdb_name = 'sunrgbd_train_rgb_i_100_8bits'
output_dir = pytorchpath+'models/'+imdb_name+'_noChairs/'

cfg_file = pytorchpath+'experiments/cfgs/faster_rcnn_end2end_sunrgbd.yml'

_DEBUG = True

# ------------



# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
print "Loading imdb..."
imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# load net
print "Creating net..."
net = FasterRCNN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)

print "Loading weight..."
# pretrained_model = pytorchpath+'data/pretrained_model/VGG_imagenet.npy'
# network.load_pretrained_npy(net, pretrained_model)
# model_file = '/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5'
# model_file = 'models/saved_model3/faster_rcnn_60000.h5'
model_file = '/home/jguerry/workspace/jg_dl/jg_pyt/models/sunrgbd_train_rgb_i_100_8bits/faster_rcnn_200000.h5'
network.load_net(model_file, net)

print "Configuring parameters..."
# exp_name = 'vgg16_02-19_13-24'
start_step = 200000
end_step = 400000
lr_decay_steps = {220000, 240000, 260000, 280000, 300000, 350000}
lr_decay = 1./10
rand_seed = 1024
lr = 0.0001
# network.weights_normal_init([net.bbox_fc, net.score_fc, net.fc6, net.fc7], dev=0.01)


disp_interval = 1000
save_interval = 20000





if rand_seed is not None:
    np.random.seed(rand_seed)
net.cuda()
net.train()

params = list(net.parameters())
# optimizer = torch.optim.Adam(params[-8:], lr=lr)
optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)



# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

print "Start training..."
myClassesDict = {}
for step in range(start_step+1, end_step+1):

    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    im_info = blobs['im_info']
    gt_boxes = blobs['gt_boxes']
    gt_ishard = blobs['gt_ishard']
    dontcare_areas = blobs['dontcare_areas']
    dontcare_areas = None

    indices = []
    for id_gt,gt in enumerate(gt_boxes):
        ranks = {key: rank for rank, key in enumerate(sorted(myClassesDict, key=myClassesDict.get, reverse=True), 1)}
        cls=int(gt[4])
        if imdb._classes[cls] not in myClassesDict.keys():
            indices.append(id_gt)
            myClassesDict[imdb._classes[cls]] = myClassesDict.get(imdb._classes[cls], 0) + 1
        elif ranks[imdb._classes[cls]]>0.5*len(myClassesDict):
            indices.append(id_gt)
            myClassesDict[imdb._classes[cls]] = myClassesDict.get(imdb._classes[cls], 0) + 1
        else:
            pass
    gt_boxes = gt_boxes[indices,:]
    gt_ishard = gt_ishard[indices]
    # forward
    if not len(gt_boxes)>0:
        continue

    # forward
    net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
    loss = net.loss + net.rpn.loss

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
            log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.rpn.cross_entropy.data.cpu().numpy()[0], net.rpn.loss_box.data.cpu().numpy()[0],
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )
        re_cnt = True

        myClasses_sorted = sorted(myClassesDict.items(), key=operator.itemgetter(1),reverse=True)
        print myClasses_sorted

    if (step % save_interval == 0) and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))
    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False

print "Training finished..."


myClasses_sorted = sorted(myClassesDict.items(), key=operator.itemgetter(1),reverse=True)
print myClasses_sorted
