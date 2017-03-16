pytorchpath = '/data02/jguerry/jg_pyt/'

import os
import torch
import cv2
import cPickle
import numpy as np

from faster_rcnn import network
from faster_rcnn.faster_rcnn_y import FasterRCNN_y
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.nms_wrapper import nms

from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file, get_output_dir


# hyper-parameters
# ------------
imdb_name_0 = 'inout_test_Images'
imdb_name_1 = 'inout_test_Depth'

cfg_file = pytorchpath+'experiments/cfgs/faster_rcnn_end2end_inout.yml'

trained_model = pytorchpath+'models/inout_y/faster_rcnn_100000.h5'

output_dir_detections = pytorchpath+'output/faster_rcnn_inout_exp/inout_test_x/detections/'

rand_seed = 1024

save_name = 'inout_x_10000'
max_per_image = 600
thresh = 0.05
vis = True
sav = True

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)


def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def im_detect(net_x, image_0, image_1):
    """Detect object classes in an image given object proposals.
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    im_data_0, im_scales_0 = net_x.get_image_blob(image_0)
    im_data_1, im_scales_1 = net_x.get_image_blob(image_1)

    im_info = np.array(
        [[im_data_0.shape[1], im_data_0.shape[2], im_scales_0[0]]],
        dtype=np.float32)


    cls_prob_0, bbox_pred_0, rois = net_x(im_data_0, im_data_1, im_info)
    scores_0 = cls_prob_0.data.cpu().numpy()
    boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas_0 = bbox_pred_0.data.cpu().numpy()
        pred_boxes_0 = bbox_transform_inv(boxes, box_deltas_0)
        pred_boxes_0 = clip_boxes(pred_boxes_0, image_0.shape)


    else:
        print "bbox reg compulsory"
        exit(1)


    return scores_0, pred_boxes_0


def test_net_y(name, net_x, imdb_0, imdb_1, max_per_image=300, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb_0.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb_0.num_classes)]

    output_dir_0 = get_output_dir(imdb_0, name)
    output_dir_1 = get_output_dir(imdb_1, name)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    # det_file_0 = os.path.join(output_dir_0, 'detections.pkl')
    # det_file_1 = os.path.join(output_dir_1, 'detections.pkl')
    det_file_x = os.path.join(output_dir_0, 'detections_x.pkl')

    for i in range(num_images):

        im_0 = cv2.imread(imdb_0.image_path_at(i))
        im_1 = cv2.imread(imdb_1.image_path_at(i))

        _t['im_detect'].tic()
        scores_0, boxes_0 = im_detect(net_x, im_0, im_1)
        detect_time = _t['im_detect'].toc(average=False)

        _t['misc'].tic()
        if vis or sav:
            # im2show = np.copy(im[:, :, (2, 1, 0)])
            im2show = np.copy(im_0)

        # skip j = 0, because it's the background class
        for j in xrange(1, imdb_0.num_classes):

            inds_0 = np.where(scores_0[:, j] > thresh)[0]
            # print inds_0.shape
            # print inds_1.shape

            cls_scores_0 = scores_0[inds_0, j]

            # print cls_scores_0.shape
            # print cls_scores_1.shape
            # print cls_scores_x.shape

            cls_boxes_0 = boxes_0[inds_0, j * 4:(j + 1) * 4]

            # print cls_boxes_0.shape
            # print cls_boxes_1.shape
            # print cls_boxes_x.shape


            keep = nms(cls_boxes_0, cfg.TEST.NMS)
            cls_boxes_0 = cls_boxes_0[keep, :]
            if vis:
                im2show = vis_detections(im2show, imdb_0.classes[j], cls_boxes_0)
            all_boxes[j][i] = cls_boxes_0

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb_0.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb_0.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc(average=False)

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time)

        # if vis:
        #     cv2.imshow('test', im2show)
        #     cv2.waitKey(1)
        if sav:
            cv2.imwrite(output_dir_detections+str(i)+'.png', im2show)

    with open(det_file_x, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb_0.evaluate_detections(all_boxes, output_dir_0)


if __name__ == '__main__':

    imdb_0 = get_imdb(imdb_name_0)
    imdb_0.competition_mode(on=True)


    imdb_1 = get_imdb(imdb_name_1)
    imdb_1.competition_mode(on=True)

    net = FasterRCNN_y(classes=imdb_1.classes, debug=False)
    network.load_net(trained_model, net)
    print('load model 1 successfully!')
    net.cuda()
    net.eval()

    # evaluation
    test_net_y(save_name, net, imdb_0, imdb_1, max_per_image, thresh=thresh, vis=vis)
