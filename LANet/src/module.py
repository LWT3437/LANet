import tensorflow as tf
from ops import *
from utils import *


def discriminator(image, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, 64, name='d_h0_conv'))
        h1 = lrelu(instance_norm(conv2d(h0, 64 * 2, name='d_h1_conv'), 'd_bn1'))
        h2 = lrelu(instance_norm(conv2d(h1, 64 * 4, name='d_h2_conv'), 'd_bn2'))
        h3 = lrelu(instance_norm(conv2d(h2, 64 * 8, name='d_h3_conv'), 'd_bn3'))
        h4 = tf.nn.sigmoid(conv2d(h3, 1, s=1, name='d_h3_pred'))

        return h4


def generator_unet(image, is_training=True, reuse=False, name="generator"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False


        resnet, skips = resnet50(image, is_training=is_training, name="resnet50")

        e6 = instance_norm(conv2d(resnet.outputs, 1024, name='g_e6_conv'), 'g_bn_e6')
        e7 = instance_norm(conv2d(tf.nn.relu(e6), 1024, name='g_e7_conv'), 'g_bn_e7')
        d6 = instance_norm(deconv2d(tf.nn.relu(e7), 1024, name='g_d6'), 'g_bn_d6')

        d5p = instance_norm(deconv2d(tf.nn.relu(d6), 512, name='g_d5p'), 'g_bn_d5p')
        s5 = instance_norm(conv2d(skips[4].outputs, 512, ks=1, s=1, name='g_d5s'), 'g_bn_d5s')
        d5 = instance_norm(deconv2d(tf.nn.relu(d5p), 512, s=1, name='g_d5'), 'g_bn_d5')
        d5 = skip_connection(d5, s5, is_training, name='g_skip_d5')

        d4p = instance_norm(deconv2d(tf.nn.relu(d5), 256, name='g_d4p'), 'g_bn_d4p')
        s4 = instance_norm(conv2d(skips[3].outputs, 256, ks=1, s=1, name='g_d4s'), 'g_bn_d4s')
        d4 = instance_norm(deconv2d(tf.nn.relu(d4p), 256, s=1, name='g_d4'), 'g_bn_d4')
        d4 = skip_connection(d4, s4, is_training, name='g_skip_d4')

        d3p = instance_norm(deconv2d(tf.nn.relu(d4), 128, name='g_d3p'), 'g_bn_d3p')
        s3 = instance_norm(conv2d(skips[2].outputs, 128, ks=1, s=1, name='g_d3s'), 'g_bn_d3s')
        d3 = instance_norm(deconv2d(tf.nn.relu(d3p), 128, s=1, name='g_d3'), 'g_bn_d3')
        d3 = skip_connection(d3, s3, is_training, name='g_skip_d3')

        d2p = instance_norm(deconv2d(tf.nn.relu(d3), 64, name='g_d2p'), 'g_bn_d2p')
        s2 = instance_norm(conv2d(skips[1].outputs, 64, ks=1, s=1, name='g_d2s'), 'g_bn_d2s')
        d2 = instance_norm(deconv2d(tf.nn.relu(d2p), 64, s=1, name='g_d2'), 'g_bn_d2')
        d2 = skip_connection(d2, s2, is_training, name='g_skip_d2')

        d1p = instance_norm(deconv2d(tf.nn.relu(d2), 64, name='g_d1p'), 'g_bn_d1p')
        s1 = instance_norm(conv2d(skips[0].outputs, 64, ks=1, s=1, name='g_d1s'), 'g_bn_d1s')
        d1 = instance_norm(deconv2d(tf.nn.relu(d1p), 64, s=1, name='g_d1'), 'g_bn_d1')
        d1 = skip_connection(d1, s1, is_training, name='g_skip_d1')

        ## attention stream
        h1 = instance_norm(deconv2d(tf.nn.relu(d1), 64, s=1, name='g_h1'), 'g_bn_h1')

        a2p = instance_norm(deconv2d(tf.nn.relu(d3), 64, name='g_a2p'), 'g_bn_a2p')
        a2 = instance_norm(deconv2d(tf.nn.relu(a2p), 64, s=1, name='g_a2'), 'g_bn_a2')
        a1p = instance_norm(deconv2d(tf.nn.relu(a2), 64, name='g_a1p'), 'g_bn_a1p')
        a1 = deconv2d(tf.nn.relu(a1p), 2, s=1, name='g_a1')

        if not is_training:
            a1 = tf.image.resize_bilinear(a1, tf.shape(h1)[1:3])
        a0 = tf.nn.sigmoid(a1, 'g_fn_a0')
        # three channels element-wise multiplications
        att1, att2 = tf.split(a0, 2, axis=3, name='att_split')
        mul1 = tf.multiply(h1, att1, name='att_mul1')
        mul2 = tf.multiply(h1, att2, name='att_mul2')
        att_out = tf.concat([mul1, mul2], axis=3)
        h0 = instance_norm(deconv2d(tf.nn.relu(att_out), 64, s=1, name='g_h0'), 'g_bn_h0')
        d0p = instance_norm(deconv2d(tf.nn.relu(h0), 64, name='g_d0p'), 'g_bn_d0p')
        d0 = deconv2d(tf.nn.relu(d0p), 64, s=1, name='g_d0')

        skip_layer = tf.maximum((image + 1) / 2, 1e-8)
        skip_layer = tf.log(skip_layer)

        out = deconv2d(tf.concat([d0, skip_layer], axis=3), 3, s=1, name='out')

        mask = deconv2d(tf.nn.relu(a1p), 3, name='g_mask')
        if not is_training:
            mask = tf.image.resize_bilinear(mask, tf.shape(out)[1:3])

        return a0, mask, out, resnet


def generator_unet_bak(image, is_training=True, reuse=False, name="generator"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        c1 = conv2d(image, 64, name='e_conv1_1')
        c1 = conv2d(tf.nn.relu(batch_norm(c1, is_training, 'e_conv1_2_bn')), 64, s=1, name='e_conv1_2')
        c1 = conv2d(tf.nn.relu(batch_norm(c1, is_training, 'e_conv1_3_bn')), 64, s=1, name='e_conv1_3')

        c2 = residual_layer(c1, 64, 3, 1, is_training, 'e_res1')
        c3 = residual_layer(c2, 128, 4, 2, is_training, 'e_res2')
        c4 = residual_layer(c3, 256, 6, 2, is_training, 'e_res3')
        c5 = residual_layer(c4, 512, 4, 2, is_training, 'e_res5')

        c5 = dilated_conv2d(tf.nn.relu(batch_norm(c5, is_training, 'e_dlt1_1_bn')), 512, ks=3, rate=1, name='e_dlt1_1')
        c5 = dilated_conv2d(tf.nn.relu(batch_norm(c5, is_training, 'e_dlt1_2_bn')), 512, ks=3, rate=2, name='e_dlt1_2')
        c5 = dilated_conv2d(tf.nn.relu(batch_norm(c5, is_training, 'e_dlt1_3_bn')), 512, ks=3, rate=3, name='e_dlt1_3')
        
        c6 = dilated_conv2d(tf.nn.relu(batch_norm(c5, is_training, 'e_dlt2_1_bn')), 512, ks=3, rate=1, name='e_dlt2_1')
        c6 = dilated_conv2d(tf.nn.relu(batch_norm(c6, is_training, 'e_dlt2_2_bn')), 512, ks=3, rate=2, name='e_dlt2_2')
        c6 = dilated_conv2d(tf.nn.relu(batch_norm(c6, is_training, 'e_dlt2_3_bn')), 512, ks=3, rate=3, name='e_dlt2_3')
        
        c7 = dilated_conv2d(tf.nn.relu(batch_norm(c6, is_training, 'e_dlt3_1_bn')), 512, ks=3, rate=1, name='e_dlt3_1')
        c7 = dilated_conv2d(tf.nn.relu(batch_norm(c7, is_training, 'e_dlt3_2_bn')), 512, ks=3, rate=2, name='e_dlt3_2')
        c7 = dilated_conv2d(tf.nn.relu(batch_norm(c7, is_training, 'e_dlt3_3_bn')), 512, ks=3, rate=3, name='e_dlt3_3')

        # HDR stream
        h1 = deconv2d(tf.nn.relu(instance_norm(c7, 'hd_d1p_bn')), 256, name='hd_d1p')
        h1 = conv2d(tf.nn.relu(instance_norm(h1, 'hd_d1_bn')), 256, s=1, name='hd_d1')
        h1 = skip_connection(instance_norm(h1, 'hd_skip1_bn1'), instance_norm(c4, 'hd_skip1_bn2'), name='hd_skip1')

        h2 = deconv2d(tf.nn.relu(instance_norm(h1, 'hd_d2p_bn')), 128, name='hd_d2p')
        h2 = conv2d(tf.nn.relu(instance_norm(h2, 'hd_d2_bn')), 128, s=1, name='hd_d2')
        h2 = skip_connection(instance_norm(h2, 'hd_skip2_bn1'), instance_norm(c3, 'hd_skip2_bn2'), name='hd_skip2')

        h3 = deconv2d(tf.nn.relu(instance_norm(h2, 'hd_d3p_bn')), 64, name='hd_d3p')
        h3 = conv2d(tf.nn.relu(instance_norm(h3, 'hd_d3_bn')), 64, s=1, name='hd_d3')
        h3 = skip_connection(instance_norm(h3, 'hd_skip3_bn1'), instance_norm(c2, 'hd_skip3_bn2'), name='hd_skip3')

        # Semantic stream
        out_msk = image

        # merge streams
        h4 = deconv2d(tf.nn.relu(instance_norm(h3, 'hd_d4p_bn')), 64, name='hd_d4p')
        h4 = conv2d(tf.nn.relu(instance_norm(h4, 'hd_d4_bn')), 3, s=1, name='hd_d4')
        out = skip_connection(h4, image, name='out')

        return out_msk, out


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

