from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
import ipdb as pdb
import keras
from keras.layers.convolutional import Conv3D
from keras import backend as K
def discriminator_image(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        #c3d_1 = Conv3D(128, (3, 3, 3), padding="valid", strides=(1, 1, 1),
        #               activation="relu", name="c3d0")(image)
        
        #c3d_2 = Conv3D(128, (3, 3, 3), padding="valid", strides=(1, 1, 1),
        #               activation="relu", name="c3d0")(c3d_1)
        
        #c3d_3 = Conv3D(128, (3, 3, 3), padding="valid", strides=(1, 1, 1),
        #               activation="relu", name="c3d0")(c3d_2)

        h0 = lrelu(conv2d(image, options.df_dim, name='dd_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='dd_h1_conv'), 'dd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='dd_h2_conv'), 'dd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='dd_h3_conv'), 'dd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = lrelu(instance_norm(conv2d(h3, options.df_dim*8, s=1, name='dd_h4_conv'), 'dd_bn4'))
        # h3 is (32 x 32 x self.df_dim*8)
        h5 = lrelu(instance_norm(conv2d(h4, options.df_dim*8, s=1, name='dd_h5_conv'), 'dd_bn5'))
        # h3 is (32 x 32 x self.df_dim*8)
        h6 = conv2d(h5, 1, s=1, name='dd_h5_pred')
        # h4 is (32 x 32 x 1)
        return h6


def discriminator_video(video, options, reuse=False, name="discriminatorVideo"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv3d(video, 64, name='dv_h0_conv'))

        h1 = lrelu(tf.contrib.layers.batch_norm(
            conv3d(h0, 128, name='dv_h1_conv'), scope='dv_bn1'))
        h2 = lrelu(tf.contrib.layers.batch_norm(
            conv3d(h1, 256, name='dv_h2_conv'), scope='dv_bn2'))
        h3 = lrelu(tf.contrib.layers.batch_norm(
            conv3d(h2, 512, name='dv_h3_conv'), scope='dv_bn3'))
        # h4 = lrelu(tf.contrib.layers.batch_norm(
        #     conv3d(h3, 256, name='dv_h4_conv'), scope='dv_bn4'))
        # h5 = lrelu(tf.contrib.layers.batch_norm(
        #     conv3d(h4, 512, name='dv_h5_conv'), scope='dv_bn5'))
        h3 = linear(tf.reshape(h3, [options.batch_size, -1]), 1, 'dv_h6_lin')
        h3 = tf.nn.sigmoid(h3)
        return h3




def generator_resnet_image(video, options, reuse=False, name="generator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_gi_c1'), name+'_gi_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_gi_c2'), name+'_gi_bn2')
            return y + x

        h0 = lrelu(conv3d(video, 64,5,5,5,1,1,1,3,3,3, name='gg_i_h0_conv'))
        h1 = lrelu(tf.contrib.layers.batch_norm(
            conv3d(h0, 128,3,3,3,2,2,2,1,1,1, name='gg_i_h1_conv3d'), scope='gg_i_bn1'))
        h2 = lrelu(tf.contrib.layers.batch_norm(
            conv3d(h1, 128,3,3,3,2,2,2,1,1,1, name='gg_i_h2_conv3d'), scope='gg_i_bn2'))
        h3 = lrelu(tf.contrib.layers.batch_norm(
            conv3d(h2, 128,3,2,2,2,1,1,0,1,1, name='gg_i_h3_conv3d'), scope='g_i_bn3'))
        h4 = lrelu(tf.contrib.layers.batch_norm(
            conv3d(h3, 128,3,2,2,2,1,1,1,1,1, name='gg_i_h3_conv4d'), scope='gg_i_bn4'))
        h4 = tf.reshape(h4,[options.batch_size,28,28,128])
            
        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3

        # define G network with 9 resnet blocks ###
        r1 = residule_block(h4, options.gf_dim*4, name='gg_i_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='gg_i_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='gg_i_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='gg_i_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='gg_i_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='gg_i_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='gg_i_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='gg_i_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='gg_i_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='gg_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'gg_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='gg_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'gg_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='gg_pred_c')
        pred = tf.nn.tanh(instance_norm(pred, 'gg_pred_bn'))

        return pred

def generator_resnet_video(image, options, reuse=False, name="generator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        def residule_block_3d(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            # y = tf.pad(x, [[0, 0], [0, 0], [p, p], [p, p], [0, 0]], "REFLECT")

            # h1 = lrelu(instance_norm(conv3d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))

            # conv3d(input_, output_dim,
            #            k_t=4, k_h=4, k_w=4, d_t=2, d_h=2, d_w=2, pad_t=1, pad_h=1, pad_w=1, stddev=0.01,
            #            name="conv3d",padding='SAME'):
            y = instance_norm(conv3d(x, dim,ks,ks,ks,s,s,s,p,p,p, name=name+'_3dc0'), name+'_3dbn0')
            # y = tf.pad(tf.nn.relu(y), [[0, 0], [0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv3d(y, dim,ks,ks,ks,s,s,s,p,p,p, name=name+'_3dc1'), name+'_3dbn1')

            return y + x
        s = options.image_size
        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3 
        c1 = tf.nn.relu(instance_norm(conv2d(image, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        c4 = tf.nn.relu(instance_norm(conv2d(c3, options.gf_dim*4, 3, 2, name='g_e4_c'), 'g_e4_bn'))
        # pdb.set_trace()
        #r1 = residule_block(c3, options.gf_dim*4, name='gv_r1_2d001')
        #r2 = residule_block(r1, options.gf_dim*4, name='gv_r2_2d002')
        #r3 = residule_block(r2, options.gf_dim*4, name='gv_r3_2d003')

        #rs0 = tf.reshape(c3, [options.batch_size, 1,  27, 27, 128],name='gv0_reshape')

        #<tf.Tensor 'MirrorPad_2:0' shape=(1, 1, 28, 28, 128) dtype=float32>
        #rs0 = tf.pad(rs0,[[0,0],[0,0],[1,0],[1,0],[0,0]],"REFLECT")
        # d0 = deconv3d(rs0, [options.batch_size, 2, 32, 32, 256],  name="dec3d0")
        # d1 = deconv3d(d0, [options.batch_size, 4, 64, 64, 128],  name="dec3d1")

        c5 = tf.reshape(c4,[options.batch_size,-1])
        lin0 = linear(c5, 512 * 2 * 2 * 1, 'g_f_h00_lin', with_w=False)
        lin0 = tf.reshape(lin0, [options.batch_size, 1, 2, 2, 512])
        lin0 = tf.nn.relu(tf.contrib.layers.batch_norm(lin0, scope='g_f_h0_lin_bn'))

        #pdb.set_trace()
        #<tf.Tensor 'generatorB2A/dec3d0/conv3d_transpose/Reshape_1:0' shape=(1, 2, 31, 31, 256) dtype=float32>
        d0 = deconv3d(lin0, [options.batch_size, 1, 4, 4, 256],  name="dec3d0")
        d0 = tf.nn.relu(tf.contrib.layers.batch_norm(d0, scope='dec3d0_bn'))

        d1 = deconv3d(d0, [options.batch_size, 1, 8, 8, 128],  name="dec3d1")
        d1 = tf.nn.relu(tf.contrib.layers.batch_norm(d1, scope='dec3d1_bn'))

        d2 = deconv3d(d1, [options.batch_size, 2, 16, 16, 64],  name="dec3d2")
        d2 = tf.nn.relu(tf.contrib.layers.batch_norm(d2, scope='dec3d2_bn'))

        d3 = deconv3d(d2, [options.batch_size, 4, 32, 32, 3],  name="dec3d3")
        d3 = tf.nn.relu(tf.contrib.layers.batch_norm(d3, scope='dec3d3_bn'))


        d4 = deconv3d(d3, [options.batch_size, 8, 64, 64, 3],  name="dec3d4")
        d4 = tf.nn.relu(tf.contrib.layers.batch_norm(d4, scope='dec3d4_bn'))

        #rc1 = residule_block_3d(d4, 3, name='gv_r1_3d')
        #rc2 = residule_block_3d(rc1, 3, name='gv_r2_3d')
        #rc3 = residule_block_3d(rc2, 3, name='gv_r3_3d')
        # rc4 = residule_block_3d(rc3, 3, name='gv_r4_3d')
        # rc5 = residule_block_3d(rc4, 3, name='gv_r5_3d')
        # rc6 = residule_block_3d(rc5, 3, name='gv_r6_3d')


        d5 = deconv3d(d4, [options.batch_size, 15, 128, 128, 3],  name="dec3d5")
        d5 = tf.nn.relu(tf.contrib.layers.batch_norm(d5, scope='dec3d5_bn'))




#        pdb.set_trace()
        #<tf.Tensor 'dec3d11/conv3d_transpose/Reshape_1:0' shape=(1, 5, 63, 63, 128) dtype=float32>
        #d1 = deconv3d(d0, 128, 3,3,3, 2,2,2,   name="dec3d1")
        #d1 = tf.pad(d1,[[0,0],[0,0],[1,0],[1,0],[0,0]],"REFLECT")
        pred_d4 = tf.nn.tanh(instance_norm(d5, 'gv_pred_bn'))
        return pred_d4

        rc1 = residule_block_3d(d4, 3, name='gv_r1_3d')
        rc2 = residule_block_3d(rc1, 3, name='gv_r2_3d')
        rc3 = residule_block_3d(rc2, 3, name='gv_r3_3d')
        rc4 = residule_block_3d(rc3, 3, name='gv_r4_3d')
        rc5 = residule_block_3d(rc4, 3, name='gv_r5_3d')
        rc6 = residule_block_3d(rc5, 3, name='gv_r6_3d')
        rc7 = residule_block_3d(rc6, 3, name='gv_r7_3d')
        rc8 = residule_block_3d(rc7, 3, name='gv_r8_3d')

        # <tf.Tensor 'generatorB2A/add_11:0' shape=(1, 5, 64, 64, 128) dtype=float32>        
        rc9 = residule_block_3d(rc8, 3 , name='gv_r9_3d')

        
        pred = tf.nn.tanh(instance_norm(rc9, 'gv_pred_bn'))
        return pred
        # what is the difference between the two things 
        # d2 = deconv3d(rc9, [options.batch_size, 8, 96, 96, 64],  name="dec3d2")
        #<tf.Tensor 'generatorB2A/Relu_6:0' shape=(1, 16, 128, 128, 64) dtype=float32>
        d2 = deconv3d(rc9, 64, 4,2,2, 3,2,2 ,  name="dec3d2")
        d2 = tf.nn.relu(instance_norm(d2, 'gv_d2_bn'))
       
        #test vimdiff 3
        

        h0 = lrelu(conv3d(d2, options.gf_dim,2,3,3,1,1,1,1,2,2, name='g_d2_conv3d_0',padding='VALID'))
        
        h1 = lrelu(conv3d(h0, options.gf_dim / 2, 1,7,7,1,1,1,0,0,0, name='g_d2_conv3d_1',padding='VALID'))
        h2 = lrelu(conv3d(h1, options.gf_dim / 4, 1,7,7,1,1,1,0,0,0, name='g_d2_conv3d_2',padding='VALID'))
        h3 = lrelu(conv3d(h2, 3, 1,3,3,1,1,1,0,0,0, name='g_d2_conv3d_3',padding='VALID'))
        # print 1/0

        # d3 = deconv3d(h1, [options.batch_size, 15, 112, 112, 3],  name="dec3d3")
        # h2 = lrelu(conv3d(d3, 3,3,3,3,1,1,1,2,2,2, name='g_d3_conv3d_0'))
        # h3 = lrelu(conv3d(h2, 3,3,3,3,1,1,1,2,2,2, name='g_d3_conv3d_1'))



        # rc10 = residule_block_3d(rc9, options.gf_dim*4, name='g_r10')
        # rc11 = residule_block_3d(rc10, options.gf_dim*4, name='g_r11')

        pred = tf.nn.tanh(instance_norm(h3, 'gv_pred_bn'))

        return pred





def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
