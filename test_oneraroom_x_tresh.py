import os
import torch
import cv2
import cPickle
import numpy as np
import errno

from faster_rcnn import network
from faster_rcnn.faster_rcnn_x import FasterRCNN, FasterRCNN_x
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.nms_wrapper import nms

from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file, get_output_dir

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# hyper-parameters
# ------------zz
pytorchpath = os.environ['PWD']+'/'


imdb_train_name_0 = 'oneraroom_2017_no_static_rgb'
imdb_train_name_1 = 'oneraroom_2017_no_static_depth_8bits'
imdb_test_name_0 = 'oneraroom_static_monotonous_rgb'
imdb_test_name_1 = 'oneraroom_static_monotonous_depth_8bits'

save_name = 'oneraroom_x_tresh_2017_no_static_on_static_monotonous_rgbd_10000'

trained_model_0 = pytorchpath+'models/'+imdb_train_name_0+'/faster_rcnn_10000.h5'
trained_model_1 = pytorchpath+'models/'+imdb_train_name_1+'/faster_rcnn_10000.h5'

output_dir = pytorchpath+'output/faster_rcnn_oneraroom_exp/'
output_dir_detections = output_dir+save_name+'/detections/'
det_file = output_dir+save_name+'/detections_'+save_name+'.pkl'

mkdir_p(output_dir_detections)



cfg_file = pytorchpath+'experiments/cfgs/faster_rcnn_end2end_oneraroom.yml'
rand_seed = 1024

max_per_image = 300
thresh = 0.5
vis = True
sav = True

# ------------

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
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (255, 0, 0), 4)
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

    im_data_0, im_scales_0 = net_x.frcnn_0.get_image_blob(image_0)
    # im_data_0=0*im_data_0
    im_data_1, im_scales_1 = net_x.frcnn_1.get_image_blob(image_1)

    im_info = np.array(
        [[im_data_0.shape[1], im_data_0.shape[2], im_scales_0[0]]],
        dtype=np.float32)


    cls_prob_0, bbox_pred_0, cls_prob_1, bbox_pred_1, rois = net_x(im_data_0, im_data_1, im_info)
    scores_0 = cls_prob_0.data.cpu().numpy()
    scores_1 = cls_prob_1.data.cpu().numpy()
    boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas_0 = bbox_pred_0.data.cpu().numpy()
        pred_boxes_0 = bbox_transform_inv(boxes, box_deltas_0)
        pred_boxes_0 = clip_boxes(pred_boxes_0, image_0.shape)

        box_deltas_1 = bbox_pred_1.data.cpu().numpy()
        pred_boxes_1 = bbox_transform_inv(boxes, box_deltas_1)
        pred_boxes_1 = clip_boxes(pred_boxes_1, image_1.shape)
    else:
        print "bbox reg compulsory"
        exit(1)


    return scores_0, scores_1, pred_boxes_0, pred_boxes_1


def test_net_x(net_x, imdb_0, imdb_1, max_per_image=300, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb_0.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb_0.num_classes)]


    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}


    for i in range(num_images):

        im_0 = cv2.imread(imdb_0.image_path_at(i))
        im_1 = cv2.imread(imdb_1.image_path_at(i))

        _t['im_detect'].tic()
        scores_0, scores_1, boxes_0, boxes_1 = im_detect(net_x, im_0, im_1)
        detect_time = _t['im_detect'].toc(average=False)




        _t['misc'].tic()

        #apply intensity treshold
        # make a keep vector
        # print boxes_0.shape
        keep_tresh_0=np.zeros((boxes_0.shape[0],1))
        for k in range(boxes_0.shape[0]):
            x1_0=boxes_0[k,0]
            y1_0=boxes_0[k,1]
            x2_0=boxes_0[k,2]
            y2_0=boxes_0[k,3]
            im_0[x1_0:x2_0,y1_0:y2_0] = 0*im_0[x1_0:x2_0,y1_0:y2_0]
            # extractedbox = im_0[x1_0:x2_0,y1_0:y2_0]
            # extractedbox = 0*extractedbox
            #compute mean of extractedbox
            # if np.mean(extractedbox)>tresh_0:
            #     keep_tresh_0[k]=1
            #change an indice in a "keep" vector
        #filter boxes thanks to "keep" vector
        #inds_tresh_0 = np.where(np.mean(im_0[boxes_0[k,0]:boxes_0[k,2],boxes_0[k,1]:boxes_0[k,3]]) > tresh_0)[0]



        if vis or sav:
            # im2show = np.copy(im[:, :, (2, 1, 0)])
            im2show = np.copy(im_0)

        # skip j = 0, because it's the background class
        for j in xrange(1, imdb_0.num_classes):

            inds_0 = np.where(scores_0[:, j] > thresh)[0]
            inds_1 = np.where(scores_1[:, j] > thresh)[0]
            # print inds_0.shape
            # print inds_1.shape

            cls_scores_0 = scores_0[inds_0, j]
            cls_scores_1 = scores_1[inds_1, j]
            cls_scores_x = np.hstack((cls_scores_0,cls_scores_1))

            # print cls_scores_0.shape
            # print cls_scores_1.shape
            # print cls_scores_x.shape

            cls_boxes_0 = boxes_0[inds_0, j * 4:(j + 1) * 4]
            cls_boxes_1 = boxes_1[inds_1, j * 4:(j + 1) * 4]
            cls_boxes_x = np.vstack((cls_boxes_0,cls_boxes_1))

            # print cls_boxes_0.shape
            # print cls_boxes_1.shape
            # print cls_boxes_x.shape

            # cls_dets_0 = np.hstack((cls_boxes_, cls_scores_0[:, np.newaxis])).astype(np.float32, copy=False)
            cls_dets_x = np.hstack((cls_boxes_x, cls_scores_x[:, np.newaxis])).astype(np.float32, copy=False)

            keep = nms(cls_dets_x, cfg.TEST.NMS)
            cls_dets_x = cls_dets_x[keep, :]
            if vis:
                im2show = vis_detections(im2show, imdb_0.classes[j], cls_dets_x)
            all_boxes[j][i] = cls_dets_x

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

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb_0.evaluate_detections(all_boxes, output_dir)


# if __name__ == '__main__':

imdb_0 = get_imdb(imdb_test_name_0)
imdb_0.competition_mode(on=True)
net_0 = FasterRCNN(classes=imdb_0.classes, debug=False)
network.load_net(trained_model_0, net_0)
print('load model 0 successfully!')
net_0.cuda()
net_0.eval()

imdb_1 = get_imdb(imdb_test_name_1)
imdb_1.competition_mode(on=True)
net_1 = FasterRCNN(classes=imdb_1.classes, debug=False)
network.load_net(trained_model_1, net_1)
print('load model 1 successfully!')
net_1.cuda()
net_1.eval()


net_x = FasterRCNN_x(classes=imdb_0.classes, debug=False)
net_x.frcnn_0 = net_0
net_x.frcnn_1 = net_1
net_x.cuda()
net_x.eval()
# evaluation
test_net_x(net_x, imdb_0, imdb_1, max_per_image, thresh=thresh, vis=vis)
