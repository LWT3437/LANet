import os
import argparse
import tensorflow as tf

from model import relativeHDR

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--gpu', dest='gpu', default='', help='GPU id to use')

# testing args
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='path of the test dataset')
parser.add_argument('--out_dir', dest='out_dir', default='./out', help='test output sample are saved here')
parser.add_argument('--test_width', dest='test_width', type=int, default=None, help='scale test images to this width')
parser.add_argument('--test_height', dest='test_height', type=int, default=None, help='scale test images to this height')

# training args
parser.add_argument('--train_dir', dest='train_dir', default='./train', help='path of the train dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# images in batch')
parser.add_argument('--crop_scale', dest='crop_scale', type=float, default=0.75, help='center crop images to this scale')
parser.add_argument('--image_width', dest='image_width', type=int, default=256, help='scale images to this width')
parser.add_argument('--image_height', dest='image_height', type=int, default=256, help='scale images to this height')
parser.add_argument('--lr', dest='lr', type=float, default=0.00004, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=20000,
                    help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=2000,
                    help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False,
                    help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--resnet_path', dest='resnet_path', default='', help='resnet50 pretrained model')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def main(_):
    if args.phase == 'train':
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if not os.path.exists(args.sample_dir):
            os.makedirs(args.sample_dir)
    else:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = relativeHDR(sess, args)
        model.train() if args.phase == 'train' else model.test()

if __name__ == '__main__':
    tf.app.run()
