import math
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim

from utils import *

p2c_indices = {}
c2p_indices = {}


def load_res_weights(session, weight_file, network):
    params = []

    if weight_file.lower().endswith('.npy'):
        npy = np.load(weight_file, encoding='latin1')
        for key, val in npy.item().items():
            params.append(val)
    else:
        print('No weights in suitable .npy format found for path ', weight_file)

    print('Assigning loaded weights..')
    tl.files.assign_params(session, params[:-2], network)

    return network


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def conv2d(input_, output_dim, ks=3, s=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)


def deconv2d(input_, output_dim, ks=3, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        sz = tf.shape(input_)
        p1 = (ks - 1) // 2
        p2 = ks // 2
        paddings = tf.constant([[0, 0], [p1, p2], [p1, p2], [0, 0]])
        resized = tf.image.resize_nearest_neighbor(input_, s * sz[1:3])
        resized = tf.pad(resized, paddings, 'SYMMETRIC')
        return slim.conv2d(resized, output_dim, ks, 1, padding='VALID', activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)


def gate_deconv2d(input_, output_dim, ks=3, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        sz = tf.shape(input_)
        p1 = (ks - 1) // 2
        p2 = ks // 2
        paddings = tf.constant([[0, 0], [p1, p2], [p1, p2], [0, 0]])
        resized = tf.image.resize_nearest_neighbor(input_, s * sz[1:3])
        resized = tf.pad(resized, paddings, 'SYMMETRIC')
        out = slim.conv2d(resized, output_dim, ks, 1, padding='VALID', activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)

        # return out
        out, att = tf.split(out, 2, axis=3)
        return out * tf.nn.sigmoid(att)


def skip_connection(input_, skip_layer, is_training, name="skip_connection"):
    with tf.variable_scope(name):
        _, _, _, sf = input_.shape.as_list()
        if not is_training:
            sh_ = tf.shape(skip_layer)
            input_ = tf.image.resize_bilinear(input_, sh_[1:3])

        # domain transformation for last skip connection
        if name == "g_skip_d0":
            skip_layer = tf.maximum((skip_layer + 1) / 2, 1e-8)
            skip_layer = tf.log(skip_layer)

        # specify weights for fusion of concatenation, so that it performs an element-wise addition
        weights = np.zeros((1, 1, 2*sf, sf))
        for i in range(sf):
            weights[0, 0, i, i] = 1
            weights[:, :, i+sf, i] = 1
        w_init = tf.constant_initializer(value=weights, dtype=tf.float32)
        b_init = tf.constant_initializer(value=0.0)

        output = tf.concat([input_, skip_layer], 3)
        return slim.conv2d(output, sf, 1, 1, padding='VALID', activation_fn=None, 
                           weights_initializer=w_init, biases_initializer=b_init)


def morphology_closing(x):
    filters = tf.constant(0.0, shape=[3,3,1])
    out = tf.nn.dilation2d(x, filters, strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')
    out = tf.nn.erosion2d(out, filters, strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')
    return out


def pano2ceiling(pano):
    height = pano.shape.as_list()[1]
    if height not in p2c_indices.keys():
        p2c_indices[height] = p2c(height)
    
    def bilinear(img):
        z_tl = tf.gather_nd(img, p2c_indices[height][0])
        z_tr = tf.gather_nd(img, p2c_indices[height][1])
        z_bl = tf.gather_nd(img, p2c_indices[height][2])
        z_br = tf.gather_nd(img, p2c_indices[height][3])
        row_args = p2c_indices[height][4]
        col_args = p2c_indices[height][5]
        return (1-row_args) * ((1-col_args) * z_tl + col_args * z_tr) + \
                   row_args * ((1-col_args) * z_bl + col_args * z_br)

    return tf.map_fn(bilinear, pano)


def ceiling2pano(ceiling, height):
    if height not in c2p_indices.keys():
        c2p_indices[height] = c2p(height)
        
    def bilinear(img):
        z_tl = tf.gather_nd(img, c2p_indices[height][0])
        z_tr = tf.gather_nd(img, c2p_indices[height][1])
        z_bl = tf.gather_nd(img, c2p_indices[height][2])
        z_br = tf.gather_nd(img, c2p_indices[height][3])
        row_args = c2p_indices[height][4]
        col_args = c2p_indices[height][5]
        return (1-row_args) * ((1-col_args) * z_tl + col_args * z_tr) + \
                   row_args * ((1-col_args) * z_bl + col_args * z_br)

    return tf.map_fn(bilinear, ceiling)
