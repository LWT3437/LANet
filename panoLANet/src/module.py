import tensorflow as tf
from ops import *
from utils import *

MASK = c2p_mask(256)
maskFlag = True


def UNet_network(image, is_training=True, reuse=False, name="UNet"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        resnet, skips = resnet50(image, is_training=is_training, name="resnet50")

        e6 = instance_norm(conv2d(resnet.outputs, 512, name='g_e6_conv'), 'g_bn_e6')
        e7 = instance_norm(conv2d(tf.nn.relu(e6), 512, name='g_e7_conv'), 'g_bn_e7')
        d6 = instance_norm(deconv2d(tf.nn.relu(e7), 512, name='g_d6'), 'g_bn_d6')

        d5p = instance_norm(deconv2d(tf.nn.relu(d6), 512, name='g_d5p'), 'g_bn_d5p')
        s5 = instance_norm(conv2d(skips[4].outputs, 512, ks=1, s=1, name='g_d5s'), 'g_bn_d5s')
        d5 = instance_norm(deconv2d(tf.nn.relu(d5p), 512, s=1, name='g_d5'), 'g_bn_d5')
        d5 = skip_connection(d5, s5, is_training, name='g_skip_d5')
        # d5 is (8 x 8 x 512)

        d4p = instance_norm(deconv2d(tf.nn.relu(d5), 256, name='g_d4p'), 'g_bn_d4p')
        s4 = instance_norm(conv2d(skips[3].outputs, 256, ks=1, s=1, name='g_d4s'), 'g_bn_d4s')
        d4 = instance_norm(deconv2d(tf.nn.relu(d4p), 256, s=1, name='g_d4'), 'g_bn_d4')
        d4 = skip_connection(d4, s4, is_training, name='g_skip_d4')
        # d4 is (16 x 16 x 256)

        d3p = instance_norm(deconv2d(tf.nn.relu(d4), 128, name='g_d3p'), 'g_bn_d3p')
        s3 = instance_norm(conv2d(skips[2].outputs, 128, ks=1, s=1, name='g_d3s'), 'g_bn_d3s')
        d3 = instance_norm(deconv2d(tf.nn.relu(d3p), 128, s=1, name='g_d3'), 'g_bn_d3')
        d3 = skip_connection(d3, s3, is_training, name='g_skip_d3')
        # d3 is (32 x 32 x 128)

        d2p = instance_norm(deconv2d(tf.nn.relu(d3), 64, name='g_d2p'), 'g_bn_d2p')
        s2 = instance_norm(conv2d(skips[1].outputs, 64, ks=1, s=1, name='g_d2s'), 'g_bn_d2s')
        d2 = instance_norm(deconv2d(tf.nn.relu(d2p), 64, s=1, name='g_d2'), 'g_bn_d2')
        d2 = skip_connection(d2, s2, is_training, name='g_skip_d2')
        # d2 is (64 x 64 x 64)

        d1p = instance_norm(deconv2d(tf.nn.relu(d2), 64, name='g_d1p'), 'g_bn_d1p')
        s1 = instance_norm(conv2d(skips[0].outputs, 64, ks=1, s=1, name='g_d1s'), 'g_bn_d1s')
        d1 = instance_norm(deconv2d(tf.nn.relu(d1p), 64, s=1, name='g_d1'), 'g_bn_d1')
        d1 = skip_connection(d1, s1, is_training, name='g_skip_d1')
        # d1 is (128 x 128 x 64)

        d0 = deconv2d(tf.nn.relu(d1), 3, name='g_d0')
        out = skip_connection(d0, image, is_training, name='g_skip_d0')

        return out, resnet, d1


def panoHDR_network(pano, is_training=True, reuse=False, name="panoHDR"):
    global maskFlag
    with tf.variable_scope(name):
        # pano is 256 x 512 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        height = pano.shape.as_list()[1]

        if maskFlag:
            MASK = c2p_mask(height)
            maskFlag = False

        thr = 0.17  # Threshold for blending
        msk = tf.reduce_mean((pano + 1) / 2, axis=3, keepdims=True)
        msk = tf.minimum(1.0, tf.maximum(0.0, msk-1.0+thr)/thr)
        # msk = tf.where(msk < 1.0-thr, tf.zeros_like(msk), tf.ones_like(msk))
        msk = morphology_closing(MASK * msk)
        msk = tf.tile(msk, [1, 1, 1, 3])    # value > 235

        pano_out, _, skip = UNet_network(pano, is_training=is_training, reuse=reuse, name="UNet")

        ceiling = pano2ceiling(pano)
        resnet, _ = resnet50(ceiling, is_training=is_training, name="resnet50")
        
        d4p = instance_norm(deconv2d(resnet.outputs, 256, name='g_d4p'), 'g_bn_d4p')
        d4 = instance_norm(deconv2d(tf.nn.relu(d4p), 256, s=1, name='g_d4'), 'g_bn_d4')
        # d4 is (16 x 16 x 256)
        d3p = instance_norm(deconv2d(tf.nn.relu(d4), 128, name='g_d3p'), 'g_bn_d3p')
        d3 = instance_norm(deconv2d(tf.nn.relu(d3p), 128, s=1, name='g_d3'), 'g_bn_d3')
        # d3 is (32 x 32 x 128)
        d2p = instance_norm(deconv2d(tf.nn.relu(d3), 64, name='g_d2p'), 'g_bn_d2p')
        d2 = instance_norm(deconv2d(tf.nn.relu(d2p), 64, s=1, name='g_d2'), 'g_bn_d2')
        # d2 is (64 x 64 x 64)
        d1p = instance_norm(deconv2d(tf.nn.relu(d2), 64, name='g_d1p'), 'g_bn_d1p')
        d1 = instance_norm(deconv2d(tf.nn.relu(d1p), 64, s=1, name='g_d1'), 'g_bn_d1')
        # d1 is (128 x 128 x 64)

        a = tf.concat([d1, pano2ceiling(skip)], axis=3)
        d0p = instance_norm(deconv2d(tf.nn.relu(a), 64, name='g_d0p'), 'g_bn_d0p')
        d0 = gate_deconv2d(tf.nn.relu(d0p), 64, s=1, name='g_d0')

        ceiling_out = deconv2d(d0, 3, s=1, name='ceiling_out')


        out = (1 - msk) * pano_out + msk * ceiling2pano(ceiling_out, height)
        # out = pano_out + msk * ceiling2pano(ceiling_out, height)

        return out, msk, ceiling, ceiling_out, pano_out


def resnet50(image, is_training=True, name="resnet50"):
        input_layer = tl.layers.InputLayer(image)
        c1 = tl.layers.PadLayer(input_layer, [[0,0], [3,3], [3,3], [0,0]], "SYMMETRIC")
        c1 = tl.layers.Conv2dLayer(c1,
                                    shape=[7, 7, 3, 64],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    W_init=tf.ones_initializer(),
                                    b_init=None)
        c1 = tl.layers.BatchNormLayer(c1,
                                    act=tf.nn.relu,
                                    is_train=is_training,
                                    beta_init=tf.zeros_initializer(),
                                    gamma_init=tf.ones_initializer())
        c2 = tl.layers.MaxPool2d(c1)
        c2 = residual_layer(c2, 64, 3, [1,1,1,1], 'res1', is_training)
        c3 = residual_layer(c2, 128, 4, [1,2,2,1], 'res2', is_training)
        c4 = residual_layer(c3, 256, 6, [1,2,2,1], 'res3', is_training)
        c5 = residual_layer(c4, 512, 3, [1,2,2,1], 'res4', is_training)

        return c5, [c1, c2, c3, c4, c5]


def residual_layer(network, ch_out, num_blocks, strides, name, is_train, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        for i in range(num_blocks):
            s = strides if i == 0 else [1,1,1,1]
            network = residual_block(network, ch_out, s, 'block%i'%i, is_train, reuse)
        return network
    

def residual_block(network, ch_out, strides, name, is_train, reuse=False):
    ch_in = network.outputs.get_shape().as_list()[-1]

    with tf.variable_scope(name, reuse=reuse):
        shortcut = network

        p = max(0, (1 - strides[1] + 1) // 2)
        network = tl.layers.PadLayer(network, [[0,0], [p,p], [p,p], [0,0]], "SYMMETRIC")
        network = tl.layers.Conv2dLayer(network,
                                        shape=[1, 1, ch_in, ch_out],
                                        strides=strides,
                                        padding='VALID',
                                        W_init=tf.ones_initializer(),
                                        b_init=None,
                                        name='conv1')
        network = tl.layers.BatchNormLayer(network,
                                        act=tf.nn.relu,
                                        is_train=is_train,
                                        beta_init=tf.zeros_initializer(),
                                        gamma_init=tf.ones_initializer(),
                                        name='bn1')
                                        
        network = tl.layers.PadLayer(network, [[0,0], [1,1], [1,1], [0,0]], "SYMMETRIC")
        network = tl.layers.Conv2dLayer(network,
                                        shape=[3, 3, ch_out, ch_out],
                                        strides=[1,1,1,1],
                                        padding='VALID',
                                        W_init=tf.ones_initializer(),
                                        b_init=None,
                                        name='conv2')
        network = tl.layers.BatchNormLayer(network,
                                        act=tf.nn.relu,
                                        is_train=is_train,
                                        beta_init=tf.zeros_initializer(),
                                        gamma_init=tf.ones_initializer(),
                                        name='bn2')

        network = tl.layers.Conv2dLayer(network,
                                        shape=[1, 1, ch_out, 4*ch_out],
                                        strides=[1,1,1,1],
                                        padding='VALID',
                                        W_init=tf.ones_initializer(),
                                        b_init=None,
                                        name='conv3')
        network = tl.layers.BatchNormLayer(network,
                                        act=tf.nn.relu,
                                        is_train=is_train,
                                        beta_init=tf.zeros_initializer(),
                                        gamma_init=tf.ones_initializer(),
                                        name='bn3')

        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if ch_in != 4*ch_out:
            p = max(0, (1 - strides[1] + 1) // 2)
            shortcut = tl.layers.PadLayer(shortcut, [[0,0], [p,p], [p,p], [0,0]], "SYMMETRIC")
            shortcut = tl.layers.Conv2dLayer(shortcut,
                                            shape=[1, 1, ch_in, 4*ch_out],
                                            strides=strides,
                                            padding='VALID',
                                            W_init=tf.ones_initializer(),
                                            b_init=None,
                                            name='short')
            shortcut = tl.layers.BatchNormLayer(shortcut,
                                        act=tf.nn.relu,
                                        is_train=is_train,
                                        beta_init=tf.zeros_initializer(),
                                        gamma_init=tf.ones_initializer(),
                                        name='short_bn')

        out = tl.layers.ElementwiseLayer([network, shortcut], tf.add, name='out')
    return out


def siMSE_criterion(in_, target, alpha):
    return tf.reduce_mean(tf.reduce_mean((in_ - target)**2, [1, 2, 3]) - alpha * tf.pow(tf.reduce_mean(in_ - target, [1, 2, 3]), 2))

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))
