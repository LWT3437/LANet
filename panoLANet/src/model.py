import os
import time
import random
import threading
import tensorflow as tf
import numpy as np
from glob import glob

from module import *
from utils import *

eps = 1e-5
BUFFER_SIZE = 64
random.seed(2020)

class panoHDR(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args

        if args.fine_turn or args.phase=='test':
            self.network = panoHDR_network
        else:
            self.network = UNet_network

        self._build_model()
        self.saver = tf.train.Saver()

        self.counter = 0
        self.train_type = ''
        self.model_name = "panoLANet.ckpt"
        self.save_freq = args.save_freq
        self.print_freq = args.print_freq


    def _build_model(self):
        if self.args.phase == 'train':
            # For single-threaded queueing of frame names
            self.path_L = tf.placeholder(tf.string)
            self.path_H = tf.placeholder(tf.string)
            self.q_frames = tf.FIFOQueue(BUFFER_SIZE, [tf.string, tf.string])
            self.enqueue_op_frames = self.q_frames.enqueue([self.path_L, self.path_H])
            self.dequeue_op_frames = self.q_frames.dequeue()

            # For multi-threaded queueing of training images
            self.input_L = tf.placeholder(tf.float32, shape=[self.args.image_height, self.args.image_width, 3])
            self.input_H = tf.placeholder(tf.float32, shape=[self.args.image_height, self.args.image_width, 3])
            self.q_train = tf.FIFOQueue(BUFFER_SIZE, [tf.float32, tf.float32], 
                                        shapes=[[self.args.image_height,self.args.image_width,3], 
                                                [self.args.image_height,self.args.image_width,3]])
            self.enqueue_op_train = self.q_train.enqueue([self.input_L, self.input_H])
            tmp_receive = self.q_train.dequeue_many(self.args.batch_size)
            tmp_split_L, tmp_split_H = tf.split(tmp_receive, 2, axis=0)
            self.receive_data = tf.concat([tf.squeeze(tmp_split_L, [0]), tf.squeeze(tmp_split_H, [0])], 3)

            self.real_data = tf.placeholder(tf.float32,
                                            [None, self.args.image_height, self.args.image_width, 6],
                                            name='real_images')
            self.real_L = self.real_data[:, :, :, :3]
            self.real_H = self.real_data[:, :, :, 3:]

            if self.args.fine_turn:
                self.fake_H, self.mask, self.ceiling, self.ceiling_out, self.pano_out = \
                                            self.network(self.real_L, reuse=False, name="panoHDR")

                self.loss_sat = tf.reduce_mean(tf.pow(self.fake_H * self.mask - self.real_H * self.mask, 2))
                self.loss_nonsat = tf.reduce_mean(tf.pow(self.fake_H * (1-self.mask) - self.real_H * (1-self.mask), 2))
                self.loss = 0.2 * self.loss_sat + 0.01 * self.loss_nonsat

                loss_sat_sum = tf.summary.scalar("train/loss_sat", self.loss_sat)
                loss_nonsat_sum = tf.summary.scalar("train/loss_nonsat", self.loss_nonsat)
                loss_sum = tf.summary.scalar("train/loss", self.loss)
                self.sum = tf.summary.merge([loss_sat_sum, loss_nonsat_sum, loss_sum])
            else:
                self.fake_H, self.resnet, _ = self.network(self.real_L, reuse=False, name="UNet")

                self.loss_l2 = tf.reduce_mean(tf.pow(self.fake_H - self.real_H, 2))
                self.loss = 1.0 * self.loss_l2

                loss_l2_sum = tf.summary.scalar("UNet/loss_l2", self.loss_l2)
                loss_sum = tf.summary.scalar("UNet/loss", self.loss)
                self.sum = tf.summary.merge([loss_l2_sum, loss_sum])

            self.t_vars = tf.trainable_variables()
   

        else:
            self.test_L = tf.placeholder(tf.float32, [None, self.args.test_height, self.args.test_width, 3], name='test_L')
            self.test_H, self.mask, self.ceiling, self.ceiling_out, self.pano_out = \
                                        self.network(self.test_L, is_training=False, reuse=False, name="panoHDR")


    def test(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        sample_files = [self.args.test_dir]
        if os.path.isdir(self.args.test_dir):
            sample_files = sorted(glob('{}/*.*'.format(self.args.test_dir)))

        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for sample_file in sample_files:
            if '9C4A' in sample_file or 'AG8A' in sample_file:
                continue
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, self.args.test_height, self.args.test_width)]
            sample_image = np.array(sample_image).astype(np.float32)
            fake_imgs = self.sess.run([self.test_H, self.mask, self.ceiling, self.ceiling_out, self.pano_out], 
                                      feed_dict={self.test_L: sample_image})

            image_paths = [os.path.join(self.args.out_dir, '{0}_out.exr'.format(os.path.basename(sample_file).split('.')[0])),
                           os.path.join(self.args.out_dir, 'Msk_{0}_out.jpg'.format(os.path.basename(sample_file).split('.')[0])),
                           os.path.join(self.args.out_dir, 'Ceil_{0}_out.jpg'.format(os.path.basename(sample_file).split('.')[0])),
                           os.path.join(self.args.out_dir, 'Ceil_{0}_out.exr'.format(os.path.basename(sample_file).split('.')[0])),
                           os.path.join(self.args.out_dir, 'Pano_{0}_out.exr'.format(os.path.basename(sample_file).split('.')[0]))]

            save_images(fake_imgs[0], [1, 1], image_paths[0])
            save_images(fake_imgs[1], [1, 1], image_paths[1])
            save_images(fake_imgs[2], [1, 1], image_paths[2])
            save_images(fake_imgs[3], [1, 1], image_paths[3])
            save_images(fake_imgs[4], [1, 1], image_paths[4])


    def train(self):
        # Load model
        if self.args.fine_turn:
            tf.train.init_from_checkpoint(self.args.checkpoint_dir, {'UNet/': 'panoHDR/UNet/'})
            tf.train.init_from_checkpoint(self.args.checkpoint_dir, {'UNet/resnet50/': 'panoHDR/resnet50/'})
            print(" [*] Load SUCCESS")
            self.train_type = 'pano'
            self.save_freq = self.save_freq // 4
            self.print_freq = self.print_freq // 4

        global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                      initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=self.args.lr, global_step=global_step, decay_steps=5000,
                                                   decay_rate=0.8)
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optim = tf.train.AdamOptimizer(learning_rate, 
                           beta1=self.args.beta1).minimize(self.loss, global_step=global_step, var_list=self.t_vars)
        self.sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        dataL = sorted(glob('{}/{}LDR/*.*'.format(self.args.train_dir, self.train_type)))
        dataH = ['{}/{}HDR/'.format(self.args.train_dir, self.train_type) \
                 + path.split('/')[-1][:-7] + '.exr' for path in dataL]
        dataLH = list(zip(dataL, dataH))

        splitPos = self.args.batch_size * 10
        trainLH = dataLH[splitPos:]
        validLH = dataLH[:splitPos]

        # Threads and thread coordinator
        coord = tf.train.Coordinator()
        thread1 = threading.Thread(target=self.enqueue_frames, args=[self.enqueue_op_frames, coord, trainLH])
        thread2 = [threading.Thread(target=self.load_and_enqueue, args=[self.enqueue_op_train, coord]) for i in range(4)]
        thread1.start()
        for t in thread2:
            t.start()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

        start_time = time.time()

        # Load resnet backbone
        if not self.args.fine_turn and self.args.resnet_path != '':
            # Load pretrained ResNet weights for encoder
            print("\n\nLoading parameters for ResNet50 layers, from '%s'..." % self.args.resnet_path)
            load_res_weights(self.sess, self.args.resnet_path, self.resnet)
            print("...done!\n") 

        for epoch in range(self.args.epoch):
            batch_idxs = len(trainLH) // self.args.batch_size

            for idx in range(0, batch_idxs):
                batch_images = self.sess.run(self.receive_data)

                # Update G network
                _, summary_str = self.sess.run([self.optim, self.sum], 
                                               feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, self.counter)

                self.counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, batch_idxs, time.time() - start_time)))

                # compute validation and sample image every print frequency
                if np.mod(self.counter, self.print_freq) == 1:
                    valid_sum = tf.Summary()
                    valid_loss_l2, valid_loss_sat, valid_loss_nonsat, valid_loss = 0, 0, 0, 0

                    it_num = len(validLH) // self.args.batch_size
                    for iv in range(0, it_num):
                        batch_val_files = validLH[iv * self.args.batch_size:(iv + 1) * self.args.batch_size]
                        batch_val_images = [load_train_data(batch_file, self.args.crop_scale, self.args.image_height, self.args.image_width)
                                             for batch_file in batch_val_files]
                        batch_val_images = np.array(batch_val_images).astype(np.float32)

                        if self.args.fine_turn:
                            tmp_value = self.sess.run([self.loss_sat, self.loss_nonsat, self.loss], 
                                                      feed_dict={self.real_data: batch_val_images})
                            valid_loss_sat += tmp_value[0]
                            valid_loss_nonsat += tmp_value[1]
                            valid_loss += tmp_value[2]
                        else:
                            tmp_value = self.sess.run([self.loss_l2, self.loss], 
                                                      feed_dict={self.real_data: batch_val_images})
                            valid_loss_l2 += tmp_value[0]
                            valid_loss += tmp_value[1]

                    if self.args.fine_turn:
                        valid_sum.value.add(tag='valid/loss_sat', simple_value=valid_loss_sat/it_num)
                        valid_sum.value.add(tag='valid/loss_nonsat', simple_value=valid_loss_nonsat/it_num)
                        self.sample_model(self.args.sample_dir, epoch, self.counter)
                    else:
                        valid_sum.value.add(tag='valid/loss_l2', simple_value=valid_loss_l2/it_num)
                    valid_sum.value.add(tag='valid/loss', simple_value=valid_loss/it_num)
                    self.writer.add_summary(valid_sum, self.counter)


                # save model every save frequency
                if np.mod(self.counter, self.save_freq) == 2:
                    self.save(self.args.checkpoint_dir, self.counter)

        # Stop threads
        print("Shutting down threads...")
        try:
            coord.request_stop()
        except Exception as e:
            print("ERROR: ", e)

        # Wait for threads to finish
        print("Waiting for threads...")
        coord.join(threads)

        self.writer.close()
        self.sess.close()


    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def sample_model(self, sample_dir, epoch, idx):
        dataL = sorted(glob('{}/*.*'.format(self.args.test_dir)))
        dataH = ['{}/{}HDR/'.format(self.args.train_dir, self.train_type) \
                 + path.split('/')[-1][:-7] + '.exr' for path in dataL]
        dataLH = list(zip(dataL, dataH))
        np.random.shuffle(dataLH) 
        batch_files = dataLH[:self.args.batch_size]
        sample_images = [load_train_data(batch_file, self.args.crop_scale, self.args.image_height, self.args.image_width)
                         for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        fake_H, mask, ceiling, ceiling_out, pano_out = \
                self.sess.run([self.fake_H, self.mask, self.ceiling, self.ceiling_out, self.pano_out], 
                              feed_dict={self.real_data: sample_images})

        if np.abs(np.sqrt(self.args.batch_size) - np.round(np.sqrt(self.args.batch_size))) < eps:
            save_width = int(np.sqrt(self.args.batch_size))
            save_height = int(np.sqrt(self.args.batch_size))
        else:
            save_width = int(np.sqrt(self.args.batch_size / 2))
            save_height = int(np.sqrt(self.args.batch_size * 2))

        if self.args.fine_turn:
            save_images(sample_images[:, :, :, :int(sample_images.shape[3] / 2)], [save_height, save_width],
                        './{}/A_real_{:02d}_{:06d}.jpg'.format(sample_dir, epoch, idx))
            save_images(sample_images[:, :, :, int(sample_images.shape[3] / 2):], [save_height, save_width],
                        './{}/B_real_{:02d}_{:06d}.exr'.format(sample_dir, epoch, idx))
            save_images(fake_H, [save_height, save_width],
                        './{}/B_fake_{:02d}_{:06d}.exr'.format(sample_dir, epoch, idx))
            save_images(mask, [save_height, save_width],
                        './{}/Msk_{:02d}_{:06d}.jpg'.format(sample_dir, epoch, idx))
            save_images(ceiling, [save_height, save_width],
                        './{}/C_input_{:02d}_{:06d}.jpg'.format(sample_dir, epoch, idx))
            save_images(ceiling_out, [save_height, save_width],
                        './{}/C_fake_{:02d}_{:06d}.exr'.format(sample_dir, epoch, idx))
            save_images(pano_out, [save_height, save_width],
                        './{}/P_fake_{:02d}_{:06d}.exr'.format(sample_dir, epoch, idx))


    # For enqueueing of frame names
    def enqueue_frames(self, enqueue_op, coord, frames):
        num_frames = len(frames)
        batch_idxs = num_frames // self.args.batch_size
        i, k = 0, 0

        np.random.shuffle(frames)
        try:
            while not coord.should_stop():
                if k >= num_frames*self.args.epoch:
                    self.sess.run(self.q_frames.close())
                    break

                if i == num_frames:
                    i = 0
                    np.random.shuffle(frames)

                fname = frames[i];
                i += 1
                k += 1
                self.sess.run(enqueue_op, feed_dict={self.path_L: fname[0], self.path_H: fname[1]})
        except tf.errors.OutOfRangeError:
            pass
        except Exception as e:
            coord.request_stop(e)

    # For multi-threaded reading and enqueueing of frames
    def load_and_enqueue(self, enqueue_op, coord):
        try:
            while not coord.should_stop():
                fname = self.sess.run(self.dequeue_op_frames)
                fname[0] = fname[0].decode('utf8')
                fname[1] = fname[1].decode('utf8')

                # Load pairs of HDR/LDR images
                real_data = load_train_data(fname, self.args.crop_scale, self.args.image_height, self.args.image_width)
                self.sess.run(enqueue_op, feed_dict={self.input_L: real_data[:,:,:3], self.input_H: real_data[:,:,3:]})
        except Exception as e:
            try:
                self.sess.run(q_train.close())
            except Exception as e:
                pass
