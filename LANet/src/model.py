import os
import time
import threading
import tensorflow as tf
import numpy as np
from glob import glob

from module import *
from utils import *

eps = 1e-5
BUFFER_SIZE = 64

class relativeHDR(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args

        self.discriminator = discriminator
        self.generator = generator_unet

        self._build_model()
        self.saver = tf.train.Saver()
        self.counter = 0


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
           
            # compute gt mask
            real_H_I = tf.reduce_mean(self.real_H, axis=3, keepdims=True)
            mask_o = morphology_closing(tf.where(real_H_I > 0.1, tf.ones_like(real_H_I), tf.zeros_like(real_H_I)))
            mask_u = morphology_closing(tf.where(real_H_I < -5.5, tf.ones_like(real_H_I), tf.zeros_like(real_H_I)))
            mask_c = 1.0 - (mask_o + mask_u)
            self.mask_r = tf.concat([mask_o, mask_c, mask_u], axis=3)
            real_H_R, real_H_G, real_H_B = tf.split(self.real_H, 3, axis=3)

            self.msk_att, self.mask, self.fake_H, self.resnet = self.generator(self.real_L, reuse=False, name="generator")
            fake_H_I = tf.reduce_mean(self.fake_H, axis=3, keepdims=True)
            fake_H_R, fake_H_G, fake_H_B = tf.split(self.fake_H, 3, axis=3)

           
            self.g_loss_l2 = siMSE_criterion(self.fake_H, self.real_H, 1.0)
            self.g_loss_mask = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.mask_r, self.mask))
            self.g_loss_color = abs_criterion(fake_H_R-fake_H_G, real_H_R-real_H_G) \
                              + abs_criterion(fake_H_G-fake_H_B, real_H_G-real_H_B) \
                              + abs_criterion(fake_H_B-fake_H_R, real_H_B-real_H_R)
            self.g_loss = 1.0 * self.g_loss_l2 + 0.05 * self.g_loss_mask# + 0.02 * self.g_loss_color

           

            g_loss_l2_sum = tf.summary.scalar("generator/g_loss_l2", self.g_loss_l2)
            g_loss_mask_sum = tf.summary.scalar("generator/g_loss_mask", self.g_loss_mask)
            g_loss_color_sum = tf.summary.scalar("generator/g_loss_color", self.g_loss_color)
            g_loss_sum = tf.summary.scalar("generator/g_loss", self.g_loss)
            self.g_sum = tf.summary.merge([g_loss_l2_sum, g_loss_mask_sum, g_loss_color_sum, g_loss_sum])

       
            t_vars = tf.trainable_variables()
            self.g_vars = [var for var in t_vars if 'generator' in var.name]
            for var in t_vars:
                print(var.name)

        else:
            self.test_L = tf.placeholder(tf.float32, [None, self.args.test_height, self.args.test_width, 3], name='test_L')
            self.msk_att, self.mask, self.test_H, _ = self.generator(self.test_L, is_training=False, reuse=False, name="generator")


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
            fake_imgs = self.sess.run([self.msk_att, self.mask, self.test_H], feed_dict={self.test_L: sample_image})

            b, h, w, _ = fake_imgs[0].shape
            msk_att = np.concatenate([fake_imgs[0], np.zeros([b, h, w, 1])], axis=3)
            mask = np.exp(fake_imgs[1])/np.sum(np.exp(fake_imgs[1]), axis=3, keepdims=True)

            msk_att_path = os.path.join(self.args.out_dir, 'Msk_att_{0}_out.jpg'.format(os.path.basename(sample_file)))
            mask_path = os.path.join(self.args.out_dir, 'Msk_{0}_out.jpg'.format(os.path.basename(sample_file)))
            image_path = os.path.join(self.args.out_dir, '{0}_out.exr'.format(os.path.basename(sample_file).split('.')[0]))
            save_images(msk_att, [1, 1], msk_att_path)
            save_images(mask, [1, 1], mask_path)
            save_images(fake_imgs[2], [1, 1], image_path)


    def train(self):
        global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                      initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=self.args.lr, global_step=global_step, decay_steps=5000,
                                                   decay_rate=0.8)
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.g_optim = tf.train.AdamOptimizer(learning_rate, 
                           beta1=self.args.beta1).minimize(self.g_loss, global_step=global_step, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        dataL = sorted(glob('{}/LDR/*.*'.format(self.args.train_dir)))
        dataH = ['{}/HDR/'.format(self.args.train_dir) \
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

        # Load model
        if self.args.resnet_path != '':
            # Load pretrained ResNet weights for encoder
            print("\n\nLoading parameters for ResNet50 layers, from '%s'..." % self.args.resnet_path)
            load_res_weights(self.sess, self.args.resnet_path, self.resnet)
            print("...done!\n") 

        if self.args.continue_train:
            if self.load(self.args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(self.args.epoch):
            batch_idxs = len(trainLH) // self.args.batch_size

            for idx in range(0, batch_idxs):
                batch_images = self.sess.run(self.receive_data)

                # Update G network
                _, summary_str = self.sess.run([self.g_optim, self.g_sum], 
                                               feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, self.counter)

                self.counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, batch_idxs, time.time() - start_time)))

                # compute validation and sample image every print frequency
                if np.mod(self.counter, self.args.print_freq) == 1:
                    g_valid_sum = tf.Summary()
                    g_valid_loss_l2, g_valid_loss_mask, g_valid_loss_color, g_valid_loss = 0, 0, 0, 0
                    
                    it_num = len(validLH) // self.args.batch_size
                    for iv in range(0, it_num):
                        batch_val_files = validLH[iv * self.args.batch_size:(iv + 1) * self.args.batch_size]
                        batch_val_images = [load_train_data(batch_file, self.args.crop_scale, self.args.image_height, self.args.image_width)
                                             for batch_file in batch_val_files]
                        batch_val_images = np.array(batch_val_images).astype(np.float32)

                        tmp_value = self.sess.run([self.g_loss_l2, self.g_loss_mask, self.g_loss_color, self.g_loss], 
                                                  feed_dict={self.real_data: batch_val_images})
                        g_valid_loss_l2 += tmp_value[0]
                        g_valid_loss_mask += tmp_value[1]
                        g_valid_loss_color += tmp_value[2]
                        g_valid_loss += tmp_value[3]

                   
                    g_valid_sum.value.add(tag='generator_valid/g_loss_l2', simple_value=g_valid_loss_l2/it_num)
                    g_valid_sum.value.add(tag='generator_valid/g_loss_mask', simple_value=g_valid_loss_mask/it_num)
                    g_valid_sum.value.add(tag='generator_valid/g_loss_color', simple_value=g_valid_loss_color/it_num)
                    g_valid_sum.value.add(tag='generator_valid/g_loss', simple_value=g_valid_loss/it_num)
                    self.writer.add_summary(g_valid_sum, self.counter)

                    
                    self.sample_model(self.args.sample_dir, epoch, self.counter)

                # save model every save frequency
                if np.mod(self.counter, self.args.save_freq) == 2:
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
        model_name = "LANet.ckpt"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
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
        dataH = ['{}/HDR/'.format(self.args.train_dir) \
                 + path.split('/')[-1][:-7] + '.exr' for path in dataL]
        dataLH = list(zip(dataL, dataH))
        np.random.shuffle(dataLH) 
        batch_files = dataLH[:self.args.batch_size]
        sample_images = [load_train_data(batch_file) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        msk_att, mask, fake_H, mask_r = self.sess.run([self.msk_att, self.mask, self.fake_H, self.mask_r], 
                                                      feed_dict={self.real_data: sample_images})

        if np.abs(np.sqrt(self.args.batch_size) - np.round(np.sqrt(self.args.batch_size))) < eps:
            save_width = int(np.sqrt(self.args.batch_size))
            save_height = int(np.sqrt(self.args.batch_size))
        else:
            save_width = int(np.sqrt(self.args.batch_size * 2))
            save_height = int(np.sqrt(self.args.batch_size / 2))

        b, h, w, _ = msk_att.shape
        mask = np.exp(mask)/np.sum(np.exp(mask), axis=3, keepdims=True)
        msk_att = np.concatenate([msk_att, np.zeros([b, h, w, 1])], axis=3)

        save_images(sample_images[:, :, :, :int(sample_images.shape[3] / 2)], [save_height, save_width],
                    './{}/A_real_{:02d}_{:06d}.jpg'.format(sample_dir, epoch, idx))
        save_images(sample_images[:, :, :, :int(sample_images.shape[3] / 2)], [save_height, save_width],
                    './{}/A_linear_{:02d}_{:06d}.exr'.format(sample_dir, epoch, idx))
        save_images(sample_images[:, :, :, int(sample_images.shape[3] / 2):], [save_height, save_width],
                    './{}/B_real_{:02d}_{:06d}.exr'.format(sample_dir, epoch, idx))
        save_images(fake_H, [save_height, save_width],
                    './{}/B_fake_{:02d}_{:06d}.exr'.format(sample_dir, epoch, idx))
        save_images(mask, [save_height, save_width],
                    './{}/Msk_{:02d}_{:06d}.jpg'.format(sample_dir, epoch, idx))
        save_images(mask_r, [save_height, save_width],
                    './{}/Msk_R_{:02d}_{:06d}.jpg'.format(sample_dir, epoch, idx))
        save_images(msk_att, [save_height, save_width],
                    './{}/Msk_att_{:02d}_{:06d}.jpg'.format(sample_dir, epoch, idx))


    def gradient_penalty(self):
        alpha = tf.random_uniform(shape=[self.args.batch_size, 1, 1, 1], minval=0., maxval=1.)
        differences = self.fake_H - self.real_H
        interpolates = self.real_H + (alpha * differences)
        gradients = tf.gradients(self.discriminator(tf.concat([interpolates, self.mask_r], axis=3), self.args.ndf, reuse=True, name='discriminator'), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return gradient_penalty


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
