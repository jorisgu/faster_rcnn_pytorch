import os
import numpy as np


from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file


# hyper-parameters
# ------------
# pytorchpath = '/data02/jguerry/jg_pyt/'
pytorchpath = '/home/jguerry/workspace/jg_dl/jg_pyt/'

imdb_name = 'inout_test_Images'
output_dir = pytorchpath+'output/faster_rcnn_inout_exp/'
correspondance_file = output_dir+'/correspondance_name_indice.txt'

cfg_file = pytorchpath+'experiments/cfgs/faster_rcnn_end2end_inout.yml'
rand_seed = 1024
if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)







def print_correspondance(imdb):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)

    with open(correspondance_file, 'wb') as f:
        for i in range(num_images):
            im_path_in = imdb.image_path_at(i)
            im_path_out = str(i)+'.png'
            f.write(im_path_in+' -> '+im_path_out+'\n')


if __name__ == '__main__':
    # load data
    imdb = get_imdb(imdb_name)
    imdb.competition_mode(on=True)

    # evaluation
    print_correspondance(imdb)
