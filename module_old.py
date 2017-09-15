from __future__ import division
import tensorflow as tf
from ops import *
#from ops_videogan import *
from utils import *
import ipdb as pdb
import keras
from keras.layers.convolutional import Conv3D
from keras import backend as K
from transformer.spatial_transformer import transformer
from transformer.tf_utils import weight_variable, bias_variable, dense_to_one_hot
	
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
		h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='dd_h1_conv'), 'dd_bn1'))
		h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='dd_h2_conv'), 'dd_bn2'))
		h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8,  name='dd_h3_conv'), 'dd_bn3'))
		h4 = lrelu(instance_norm(conv2d(h3, options.df_dim*8,  name='dd_h4_conv'), 'dd_bn4'))
		h5 = lrelu(instance_norm(conv2d(h4, options.df_dim*8,  name='dd_h5_conv'), 'dd_bn5'))
		h6 = conv2d(h5, 1, s=1, name='dd_h5_pred')
		return h6


def compress_input2vec_st( image, options, reuse=False, name='generatorA'):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False
		
		h0 = lrelu(conv2d(image, options.df_dim, name='ci_h0_conv_st'))
		h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='ci_h1_conv_st'), 'ci_bn1_st'))
		h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='ci_h2_conv_st'), 'ci_bn2_st'))
		h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, name='ci_h3_conv_st'), 'ci_bn3_st'))
		h4 = lrelu(instance_norm(conv2d(h3, options.df_dim*8, name='ci_h4_conv_st'), 'ci_bn4_st'))
		h5 = lrelu(instance_norm(conv2d(h4, options.df_dim*8, name='ci_h5_conv_st'), 'ci_bn5_st'))
		h6 = lrelu(instance_norm(conv2d(h5, options.df_dim*8, name='ci_h6_conv_st'), 'ci_bn6_st'))
		h7 = conv2d(h6, options.df_dim*8, name='cii_h6_pred_st')
		return h7



def compress_input2vec_om(image, options, reuse=False, name='generatorA'):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False
		
		h0 = lrelu(conv2d(image, options.df_dim, name='ci_h0_conv_om'))
		h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='ci_h1_conv_om'), 'ci_bn1_om'))
		h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='ci_h2_conv_om'), 'ci_bn2_om'))
		h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*4, name='ci_h3_conv_om'), 'ci_bn3_om'))
		h4 = lrelu(instance_norm(conv2d(h3, options.df_dim*4, name='ci_h4_conv_om'), 'ci_bn4_om'))
		#h5 = lrelu(instance_norm(conv2d(h4, options.df_dim*8, name='ci_h5_conv_om'), 'ci_bn5_om'))
		#h6 = lrelu(instance_norm(conv2d(h5, options.df_dim*8, name='ci_h6_conv_om'), 'ci_bn6_om'))
		h7 = conv2d(h4, options.df_dim*8, name='cii_h6_pred_om')
		return h7




def st_camera(image, video,options,reuse=False,name="generatorA"):

	with tf.variable_scope(name):
		if reuse:
			 tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False

		
		h0 = lrelu(conv2d(image, options.df_dim, name='ci_h0_conv_st'))
		h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='ci_h1_conv_st'), 'ci_bn1_st'))
		h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='ci_h2_conv_st'), 'ci_bn2_st'))
		h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, name='ci_h3_conv_st'), 'ci_bn3_st'))
		h4 = lrelu(instance_norm(conv2d(h3, options.df_dim*8, name='ci_h4_conv_st'), 'ci_bn4_st'))
		h5 = lrelu(instance_norm(conv2d(h4, options.df_dim*8, name='ci_h5_conv_st'), 'ci_bn5_st'))
		h6 = lrelu(instance_norm(conv2d(h5, options.df_dim*8, name='ci_h6_conv_st'), 'ci_bn6_st'))
		h7 = tf.reshape(conv2d(h6, options.df_dim*8, name='cii_h6_pred_st'),[1,512])

		
		W1_fc_loc1 = weight_variable([512, 6])
		W2_fc_loc1 = weight_variable([512, 6])
		W3_fc_loc1 = weight_variable([512, 6])
		W4_fc_loc1 = weight_variable([512, 6])
		#
		tmp_s = 1
		#tmp_t = 0
		#initial1 = np.array([[tmp_s,  tmp_t]]).astype('float32').flatten()
		#initial2 = np.array([[tmp_s,  tmp_t]]).astype('float32').flatten()
		#initial3 = np.array([[tmp_s,  tmp_t]]).astype('float32').flatten()
		#initial4 = np.array([[tmp_s,  tmp_t]]).astype('float32').flatten()
		

		
		#b1_fc_loc1 = bias_variable([256])
		#b2_fc_loc1 = bias_variable([256])
		#b3_fc_loc1 = bias_variable([256])
		#b4_fc_loc1 = bias_variable([256])
		
		#W1_fc_loc2 = weight_variable([256, 6])
		#W2_fc_loc2 = weight_variable([256, 6])
		#W3_fc_loc2 = weight_variable([256, 6])
		#W4_fc_loc2 = weight_variable([256, 6])
 
		initial1 = np.array([[tmp_s, 0, 0], [0, tmp_s, 0]]).astype('float32').flatten()
		initial2 = np.array([[tmp_s, 0, 0], [0, tmp_s, 0]]).astype('float32').flatten()
		initial3 = np.array([[tmp_s, 0, 0], [0, tmp_s, 0]]).astype('float32').flatten()
		initial4 = np.array([[tmp_s, 0, 0], [0, tmp_s, 0]]).astype('float32').flatten()
		
		b1_fc_loc1 = tf.Variable(initial_value=initial1, name='b1_fc_loc2')
		b2_fc_loc1 = tf.Variable(initial_value=initial2, name='b2_fc_loc2')
		b3_fc_loc1 = tf.Variable(initial_value=initial3, name='b3_fc_loc2')
		b4_fc_loc1 = tf.Variable(initial_value=initial4, name='b4_fc_loc2')
		


		# %% Define the two layer localisation network
		h1_fc_loc2 = tf.nn.tanh(tf.matmul(h7, W1_fc_loc1) + b1_fc_loc1)
		h2_fc_loc2 = tf.nn.tanh(tf.matmul(h7, W2_fc_loc1) + b2_fc_loc1)
		h3_fc_loc2 = tf.nn.tanh(tf.matmul(h7, W3_fc_loc1) + b3_fc_loc1)
		h4_fc_loc2 = tf.nn.tanh(tf.matmul(h7, W4_fc_loc1) + b4_fc_loc1)
		#zo = tf.Variable([0.0])
		#h1_fc_loc2 = tf.concat([zo,[h1_fc_loc1[0][0]],[h1_fc_loc1[0][1]],[h1_fc_loc1[0][0]],zo,[h1_fc_loc1[0][1]] ],axis=-1) 
		#
		#h2_fc_loc2 = tf.concat([zo,[h2_fc_loc1[0][0]],[h2_fc_loc1[0][1]],[h2_fc_loc1[0][0]],zo,[h2_fc_loc1[0][1]] ],axis=-1) 
		#h3_fc_loc2 = tf.concat([zo,[h3_fc_loc1[0][0]],[h3_fc_loc1[0][1]],[h3_fc_loc1[0][0]],zo,[h3_fc_loc1[0][1]] ],axis=-1) 
		#h4_fc_loc2 = tf.concat([zo,[h4_fc_loc1[0][0]],[h4_fc_loc1[0][1]],[h4_fc_loc1[0][0]],zo,[h4_fc_loc1[0][1]] ],axis=-1) 
		#h1_fc_loc2 = tf.expand_dims(h1_fc_loc2,axis=0) 
		#h2_fc_loc2 = tf.expand_dims(h2_fc_loc2,axis=0) 
		#h3_fc_loc2 = tf.expand_dims(h3_fc_loc2,axis=0) 
		#h4_fc_loc2 = tf.expand_dims(h4_fc_loc2,axis=0) 
	   
		#h1_fc_loc2 = tf.Variable(initial_value=np.array([0, 0, 0, 0, 0, 0]))
		#h2_fc_loc2 = tf.Variable(initial_value=np.array([0, 0, 0, 0, 0, 0]))
		#h3_fc_loc2 = tf.Variable(initial_value=np.array([0, 0, 0, 0, 0, 0]))
		#h4_fc_loc2 = tf.Variable(initial_value=np.array([0, 0, 0, 0, 0, 0]))

		#h1_fc_loc2[1:3] = h1_fc_loc1[0:2]
		#h2_fc_loc2[1:3] = h2_fc_loc1[0:2]
		#h3_fc_loc2[1:3] = h3_fc_loc1[0:2]
		#h4_fc_loc2[1:3] = h4_fc_loc1[0:2]


		#h1_fc_loc2[3] = h1_fc_loc1[2]
		#h2_fc_loc2[3] = h2_fc_loc1[2]
		#h3_fc_loc2[3] = h3_fc_loc1[2]
		#h4_fc_loc2[3] = h4_fc_loc1[2]

		#
		#h1_fc_loc2[5] = h1_fc_loc1[3]
		#h2_fc_loc2[5] = h2_fc_loc1[3]
		#h3_fc_loc2[5] = h3_fc_loc1[3]
		#h4_fc_loc2[5] = h4_fc_loc1[3]

		#h1_fc_loc1([1:2]) = 
		
		#keep_prob = tf.placeholder(tf.float32)
		#keep2_prob = tf.placeholder(tf.float32)
		#keep3_prob = tf.placeholder(tf.float32)
		#keep4_prob = tf.placeholder(tf.float32)

		

		#h1_fc_loc1_drop = tf.nn.dropout(h1_fc_loc1, keep_prob)
		#h2_fc_loc1_drop = tf.nn.dropout(h2_fc_loc1, keep_prob)
		#h3_fc_loc1_drop = tf.nn.dropout(h3_fc_loc1, keep_prob)
		#h4_fc_loc1_drop = tf.nn.dropout(h4_fc_loc1, keep_prob)


		# %% Second layer
		#h1_fc_loc2 = tf.nn.tanh(tf.matmul(h1_fc_loc1, W1_fc_loc2) + b1_fc_loc2)
		#h2_fc_loc2 = tf.nn.tanh(tf.matmul(h2_fc_loc1, W2_fc_loc2) + b2_fc_loc2)
		#h3_fc_loc2 = tf.nn.tanh(tf.matmul(h3_fc_loc1, W3_fc_loc2) + b3_fc_loc2)
		#h4_fc_loc2 = tf.nn.tanh(tf.matmul(h4_fc_loc1, W4_fc_loc2) + b4_fc_loc2)

		# %% We'll create a spatial transformer module to identify discriminative
		# %% patches
		out_size = (256, 256)
		h1_trans = transformer(tf.expand_dims(video[0][0],0), h1_fc_loc2, out_size)
		h2_trans = transformer(tf.expand_dims(video[0][1],0), h2_fc_loc2, out_size)
		h3_trans = transformer(tf.expand_dims(video[0][2],0), h3_fc_loc2, out_size)
		h4_trans = transformer(tf.expand_dims(video[0][3],0), h4_fc_loc2, out_size)
		h1_trans = tf.reshape(h1_trans,[tf.shape(h1_trans)[0],256,256,3])
		h2_trans = tf.reshape(h2_trans,[tf.shape(h2_trans)[0],256,256,3])
		h3_trans = tf.reshape(h3_trans,[tf.shape(h3_trans)[0],256,256,3])
		h4_trans = tf.reshape(h4_trans,[tf.shape(h4_trans)[0],256,256,3])
		h_all_rt = tf.expand_dims(tf.concat([h1_trans,h2_trans,h3_trans,h4_trans],0 ), 0)
		return h_all_rt




def discriminator_video(video0,image, options, reuse=False, name="discriminatorVideo"):

	with tf.variable_scope(name):
		# image is 256 x 256 x input_c_dim
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False


		image = tf.expand_dims(image,1)
		image = tf.reshape(image,[tf.shape(video0)[0],2,options.image_size,options.image_size,3])
		video = tf.concat([video0,image],1)

		h0 = lrelu(conv3d(video, 32,2,9,9,1,3,3, name='dv_h0_conv'))

		h1 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h0, 64,2,7,7,1,3,3, name='dv_h1_conv'), scope='dv_bn1'))
		h2 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h1, 128,2,5,5,1,2,2, name='dv_h2_conv'), scope='dv_bn2'))
		h3 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h2, 128,4,3,3,1,2,2, name='dv_h3_conv'), scope='dv_bn3'))

		h4 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h3, 256,4,3,3,1,2,2, name='dv_h4_conv'), scope='dv_bn4'))
		

		h5 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h4, 256,4,3,3,1,2,2, name='dv_h6_conv'), scope='dv_bn6'))

		h_tmp = tf.contrib.layers.flatten(h5)
		h7 = linear( h_tmp , 10, 'dv_h6_lin')
		#h7 = linear(h7,1,'dv_h7_lin')
		h8 = tf.nn.sigmoid(h7)

		return h8,h5
def g_background(self, h0, z, options,reuse=False,name="generatorA"):
	with tf.variable_scope(name):
	# image is 256 x 256 x input_c_dim
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False

		#e9 = tf.reshape(h0,(1,1,1,-1))
		e9 = tf.expand_dims(tf.expand_dims(h0,1),1)

		# e8 is (1 x 1 x self.gf_dim*8)
		d1 = deconv2d(tf.nn.relu(e9), options.gf_dim*8, name='g_d1')
		d1 = instance_norm(d1, 'g_bn_d1')
		# d1 is (2 x 2 x self.gf_dim*8*2)

		d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
		d2 = instance_norm(d2, 'g_bn_d2')
		# d2 is (4 x 4 x self.gf_dim*8*2)

		d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
		d3 = instance_norm(d3, 'g_bn_d3')
		# d3 is (8 x 8 x self.gf_dim*8*2)

		d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
		d4 = instance_norm(d4, 'g_bn_d4')
		# d4 is (16 x 16 x self.gf_dim*8*2)

		d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*8, name='g_d5')
		d5 = instance_norm(d5, 'g_bn_d5')
		# d5 is (32 x 32 x self.gf_dim*4*2)

		d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*8, name='g_d6')
		d6 = instance_norm(d6, 'g_bn_d6')
		# d6 is (64 x 64 x self.gf_dim*2*2)

		d7 = deconv2d(tf.nn.relu(d6), options.gf_dim*8, name='g_d7')
		d7 = instance_norm(d7, 'g_bn_d7')
		# d7 is (128 x 128 x self.gf_dim*1*2)

		d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
		# d8 is (256 x 256 x output_c_dim)

		return tf.nn.tanh(d8)



def sp_camera(image,z,options,reuse=False,name="sp_camera"):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False

		#h0 = lrelu(conv2d(image, options.df_dim, name='ci_h0_conv_sp'))
		#h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*1,7,7, name='ci_h1_conv_sp'), 'ci_bn1_sp'))
		#h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*2,7,7, name='ci_h2_conv_sp'), 'ci_bn2_sp'))
		#h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*4,6,6, name='ci_h3_conv_sp'), 'ci_bn3_sp'))
		#h4 = lrelu(instance_norm(conv2d(h3, options.df_dim*4,6,6, name='ci_h4_conv_sp'), 'ci_bn4_sp'))
		#h5 = lrelu(instance_norm(conv2d(h4, options.df_dim*8, name='ci_h5_conv_sp'), 'ci_bn5_sp'))
	   


		h0 = conv2d(image, options.df_dim, name='ci_h0_conv_sp')
		h1 = instance_norm(conv2d(lrelu(h0), options.df_dim / 2.0, 11, 4,name='ci_h1_conv_sp'), 'ci_bn1_sp')
		h2 = instance_norm(conv2d(lrelu(h1), options.df_dim / 2.0, 7,2, name='ci_h2_conv_sp'), 'ci_bn2_sp')
		h3 = instance_norm(conv2d(lrelu(h2), options.df_dim,7,2, name='ci_h3_conv_sp'), 'ci_bn3_sp')
		h4 = instance_norm(conv2d(lrelu(h3), options.df_dim,5,2, name='ci_h4_conv_sp'), 'ci_bn4_sp')
		h5 = instance_norm(conv2d(lrelu(h4), options.df_dim*2,5,2, name='ci_h5_conv_sp'), 'ci_bn5_sp')
		h6 = instance_norm(conv2d(lrelu(h5), options.df_dim*2,3,2, name='ci_h6_conv_sp'), 'ci_bn6_sp')
				
		h6 = tf.reshape(h6,(tf.shape(h6)[0],options.df_dim*2))
		#h7 = tf.concat((h6),axis=-1)
		#h7 = linear(tf.contrib.layers.flatten(h5), 128, 'ci_lin0_om', with_w=False)
		W1_fc_loc1 = weight_variable([128, 3])
		tmp_s = 0
		tmp_t = 0

		initial1 = np.array([[tmp_s, tmp_t , tmp_t]]).astype('float32').flatten()
		b1_fc_loc1 = tf.Variable(initial_value=initial1, name='b1_fc_loc2_sp')
		feat = tf.nn.tanh(tf.matmul(h6, W1_fc_loc1) + b1_fc_loc1)
		#feat = tf.clip_by_value(feat,0,0.3)

		h1_fc_loc2 = tf.multiply(feat, tf.constant(1.0))

		tf_s1 = tf.expand_dims(tf.add(h1_fc_loc2[:,0], tf.constant(0.76),name='sp_s1'),-1)
		tf_tx1 = tf.expand_dims(tf.add(h1_fc_loc2[:,1], tf.constant(0.0),name='sp_tx1'),-1)
		tf_ty1 = tf.expand_dims(tf.add(h1_fc_loc2[:,2], tf.constant(0.0),name='sp_ty1'),-1)
		mtx1 = tf.concat([tf_s1,tf.zeros([tf.shape(h6)[0],1]),tf_tx1,tf.zeros([tf.shape(h6)[0],1]),tf_s1,tf_ty1],axis=1)
		

		tf_s2 = tf.expand_dims(tf.add(tf.multiply(h1_fc_loc2[:,0],tf.constant(2.0)), tf.constant(0.76),name='sp_s2'),-1)
		tf_tx2 = tf.expand_dims(tf.add(tf.multiply(h1_fc_loc2[:,1],tf.constant(2.0)) , tf.constant(0.0),name='sp_tx2'),-1)
		tf_ty2 = tf.expand_dims(tf.add(tf.multiply(h1_fc_loc2[:,2],tf.constant(2.0)), tf.constant(0.0),name='sp_ty2'),-1)
		mtx2 = tf.concat([tf_s2,tf.zeros([tf.shape(h6)[0],1]),tf_tx2,tf.zeros([tf.shape(h6)[0],1]),tf_s2,tf_ty2],axis=1)
	  
	  
		tf_s3 = tf.expand_dims(tf.add(tf.multiply(h1_fc_loc2[:,0],tf.constant(3.0)), tf.constant(0.76),name='sp_s3'),-1)
		tf_tx3 = tf.expand_dims(tf.add(tf.multiply(h1_fc_loc2[:,1],tf.constant(3.0)) , tf.constant(0.0),name='sp_tx3'),-1)
		tf_ty3 = tf.expand_dims(tf.add(tf.multiply(h1_fc_loc2[:,2],tf.constant(3.0)), tf.constant(0.0),name='sp_ty3'),-1)
		mtx3 = tf.concat([tf_s3,tf.zeros([tf.shape(h6)[0],1]),tf_tx3,tf.zeros([tf.shape(h6)[0],1]),tf_s3,tf_ty3],axis=1)
		
		tf_s4 = tf.expand_dims(tf.add(tf.multiply(h1_fc_loc2[:,0],tf.constant(4.0)), tf.constant(0.76),name='sp_s4'),-1)
		tf_tx4 = tf.expand_dims(tf.add(tf.multiply(h1_fc_loc2[:,1],tf.constant(4.0)) , tf.constant(0.0),name='sp_tx4'),-1)
		tf_ty4 = tf.expand_dims(tf.add(tf.multiply(h1_fc_loc2[:,2],tf.constant(4.0)), tf.constant(0.0),name='sp_ty4'),-1)
		mtx4 = tf.concat([tf_s4,tf.zeros([tf.shape(h6)[0],1]),tf_tx4,tf.zeros([tf.shape(h6)[0],1]),tf_s4,tf_ty4],axis=1)


		out_size = (256, 256)
		h1_trans = transformer(image[:,:,:,0:3], mtx1, out_size)
		h2_trans = transformer(image[:,:,:,0:3], mtx2, out_size)
		h3_trans = transformer(image[:,:,:,0:3], mtx3, out_size)
		h4_trans = transformer(image[:,:,:,0:3], mtx4, out_size)
		h1_trans = tf.reshape(h1_trans,[1,256,256,3])
		h2_trans = tf.reshape(h2_trans,[1,256,256,3])
		h3_trans = tf.reshape(h3_trans,[1,256,256,3])
		h4_trans = tf.reshape(h4_trans,[1,256,256,3])
		h_all_rt = tf.expand_dims(tf.concat([h1_trans,h2_trans,h3_trans,h4_trans],0 ), 0)
		return feat,mtx1,mtx2,mtx3,mtx4,h_all_rt




		



def camera_movement_generator(self, image,z, options, reuse=False, name="sp_camera"):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False
		
		feat,mtx1,mtx2,mtx3,mtx4,video_camera = sp_camera(image,z, options=options,reuse=reuse,name=name)
		return feat,mtx1,mtx2,mtx3,mtx4,video_camera





def videogan_generator(self, image,z, options, reuse= False, name="generatorA"):
	with tf.variable_scope(name):
		# image is 256 x 256 x input_c_dim
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False        

		#first get the camera parameters 
		camera_feat,mtx1,mtx2,mtx3,mtx4,cm = camera_movement_generator(self, image,z, options, reuse=reuse, name="generatorA")

		h0 = lrelu(conv2d(image, options.df_dim, name='ci_h0_conv_om'))
		h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*1.0, 7, 4, name='ci_h1_conv_om'), 'ci_bnh1_om'))
		h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*1.0, 7, 2, name='ci_h2_conv_om'), 'ci_bnh2_om'))
		h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*2, 6, 2,name='ci_h3_conv_om'), 'ci_bnh3_om'))
		h4 = lrelu(instance_norm(conv2d(h3, options.df_dim*2, 6, 2,name='ci_h4_conv_om'), 'ci_bnh4_om'))
		#h5 = lrelu(instance_norm(conv2d(h4, options.df_dim*4, 5, 2,name='ci_h5_conv_om'), 'ci_bnh5_om'))
		h6 = instance_norm(conv2d(h4, options.df_dim*2, 5, 2,name='ci_h6_conv_om'), 'ci_bnh6_om')

		lin0 = linear(tf.contrib.layers.flatten(h6), 256, 'ch_lin0_om', with_w=False)


		i0 = lrelu(conv2d(image, options.df_dim, name='ci_i0_conv_om'))
		i1 = lrelu(instance_norm(conv2d(i0, options.df_dim*1.0, 7, 4, name='ci_i1_conv_om'), 'ci_bni1_om'))
		i2 = lrelu(instance_norm(conv2d(i1, options.df_dim*1.0, 7, 2, name='ci_i2_conv_om'), 'ci_bni2_om'))
		i3 = lrelu(instance_norm(conv2d(i2, options.df_dim*2, 6, 2,name='ci_i3_conv_om'), 'ci_bni3_om'))
		i4 = lrelu(instance_norm(conv2d(i3, options.df_dim*2, 6, 2,name='ci_i4_conv_om'), 'ci_bni4_om'))
		#i5 = lrelu(instance_norm(conv2d(h4, options.df_dim*2, 5, 2,name='ci_i5_conv_om'), 'ci_bni5_om'))
		i6 = instance_norm(conv2d(i4, options.df_dim*4, 5, 2,name='ci_i6_conv_om'), 'ci_bni6_om')
		lin1 = linear(tf.contrib.layers.flatten(i6), 256, 'ci_lin0_om', with_w=False)

		#h7 = conv2d(h6, options.df_dim*8, name='cii_h6_pred_om')
		#h8 = tf.concat([tf.contrib.layers.flatten(h6) ,camera_feat],axis=-1)
		h8 = lin0
		i8 = lin0


		gf4, mask = g_foreground(self,h8,z,camera_feat,options,reuse,name)
		
		gb4 = g_background(self,i8,z,options, reuse, name)

		gb4 = tf.reshape(gb4, [-1, 1, 256, 256, 3])
		
		gb4 = tf.tile(gb4, [1, 4, 1, 1, 1])

		static_video =  (1 - mask) * gb4 + mask * gf4
		#return cm,static_video
		tmp_flag = 0
		if tmp_flag == 1:
			mtx1 = tf.tile(tf.constant(np.array([[0.76,0,0,0,0.76,0]])),[tf.shape(image)[0],1] )
			mtx2 = tf.tile(tf.constant(np.array([[0.76,0,0,0,0.76,0]])),[tf.shape(image)[0],1] )
			mtx3 = tf.tile(tf.constant(np.array([[0.76,0,0,0,0.76,0]])),[tf.shape(image)[0],1] )
			mtx4 = tf.tile(tf.constant(np.array([[0.76,0,0,0,0.76,0]])),[tf.shape(image)[0],1] )

		out_size = (256, 256)

		h1_trans = transformer(static_video[:,0,:,:], mtx1, out_size)
		h2_trans = transformer(static_video[:,1,:,:], mtx2, out_size)
		h3_trans = transformer(static_video[:,2,:,:], mtx3, out_size)
		h4_trans = transformer(static_video[:,3,:,:], mtx4, out_size)


		h1_trans = tf.reshape(h1_trans,[tf.shape(image)[0],1,256,256,3])
		h2_trans = tf.reshape(h2_trans,[tf.shape(image)[0],1,256,256,3])
		h3_trans = tf.reshape(h3_trans,[tf.shape(image)[0],1,256,256,3])
		h4_trans = tf.reshape(h4_trans,[tf.shape(image)[0],1,256,256,3])
		
		h_all_rt = tf.concat([h1_trans,h2_trans,h3_trans,h4_trans],1 )

 
		#video_full = st_camera(image, static_video,options = options, reuse=reuse,name=name) 
		return cm, h_all_rt





def g_foreground(self, h0,z, camera_feat, options, reuse=False,name="generatorA"):
	z = None
	with tf.variable_scope(name):
		# image is 256 x 256 x input_c_dim
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False

		#h0 = tf.reshape(h0,(1,-1))
		h0 = tf.contrib.layers.flatten(h0) 

		l8, self.h0_w, self.h0_b = linear(h0, 256 * 4 * 4 * 1, 'g_f_h0_lin', with_w=True)

		h0 = tf.reshape(l8, [-1, 1, 4, 4, 256])
		h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, scope='g_f_bn0'))
 

		h1, self.h1_w, self.h1_b = deconv3d(h0,
														 [tf.shape(h0)[0], 2, 8, 8, 512],4,4,4,2,2,2,1,1,1, name='g_f_h1', with_w=True)

		h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, scope='g_f_bn1'))

		h1_1, self.h1_w_1, self.h1_b_1 = deconv3d(h1,
														 [tf.shape(h0)[0], 2, 8, 8, 512],4,4,4,1,1,1,1,1,1, name='g_f_h1_1', with_w=True)
		

		h1_1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1_1, scope='g_f_bn1_1'))

		h2, self.h2_w, self.h2_b = deconv3d(h1,
													[tf.shape(h0)[0], 4, 16, 16, 512],4,4,4,2,2,2,1,1,1, name='g_f_h2', with_w=True)
		h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, scope='g_f_bn2'))

		h2_1, self.h2_w_1, self.h2_b_1 = deconv3d(h2,
													[tf.shape(h0)[0], 4, 16, 16, 512],4,4,4,1,1,1,1,1,1, name='g_f_h2_1', with_w=True)
		h2_1 = tf.nn.relu(tf.contrib.layers.batch_norm(h2_1, scope='g_f_bn2_1'))

		#h2_1, self.h2_w_1, self.h2_b_1 = deconv3d(h2_1,
		#											[tf.shape(h0)[0], 4, 16, 16, 256],4,4,4,1,1,1,1,1,1, name='g_f_h2_2', with_w=True)
		#h2_1 = tf.nn.relu(tf.contrib.layers.batch_norm(h2_1, scope='g_f_bn2_2'))

		
		#return h2,None

		h3, self.h3_w, self.h3_b = deconv3d(h2_1,
													[tf.shape(h0)[0], 4, 32, 32, 128],4,4,4,1,2,2,1,1,1, name='g_f_h3', with_w=True)
		h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, scope='g_f_bn3'))

		h3_1, self.h3_w_1, self.h3_b_1 = deconv3d(h3,
													[tf.shape(h0)[0], 4, 32, 32, 128],4,4,4,1,1,1,1,1,1, name='g_f_h3_1', with_w=True)
		h3_1 = tf.nn.relu(tf.contrib.layers.batch_norm(h3_1, scope='g_f_bn3_1'))

		#mask1 = deconv3d(h3,[-1, 1, 32, 32, 1], name='g_mask1', with_w=False)
		#mask = l1Penalty(tf.nn.sigmoid(mask1))

		#return tf.nn.tanh(h3), mask
		h4, self.h4_w, self.h4_b = deconv3d(h3_1,
													[tf.shape(h0)[0], 4, 64, 64, 128],4,4,4,1,2,2,1,1,1, name='g_f_h4', with_w=True)

		h4 = tf.nn.relu(tf.contrib.layers.batch_norm(h4, scope='g_f_bn4'))

		h4_1, self.h4_w_1, self.h4_b_1 = deconv3d(h4,
													[tf.shape(h0)[0], 4, 64, 64, 128],4,4,4,1,1,1,1,1,1, name='g_f_h4_1', with_w=True)

		h4_1 = tf.nn.relu(tf.contrib.layers.batch_norm(h4_1, scope='g_f_bn4_1'))

		#h4_1, self.h4_w_1, self.h4_b_1 = deconv3d(h4_1,
		#											[tf.shape(h0)[0], 4, 64, 64, 64],4,4,4,1,1,1,1,1,1, name='g_f_h4_2', with_w=True)

		#h4_1 = tf.nn.relu(tf.contrib.layers.batch_norm(h4_1, scope='g_f_bn4_2'))


		h5, self.h5_w, self.h5_wb = deconv3d(h4_1,
													[tf.shape(h0)[0],4, 128, 128, 128],4,4,4,1,2,2,1,1,1, name='g_f_h5', with_w=True)

		h5 = tf.nn.relu(tf.contrib.layers.batch_norm(h5, scope='g_f_bn5'))

		h5_1, self.h5_w_1, self.h5_wb_1 = deconv3d(h5,
													[tf.shape(h0)[0],4, 128, 128, 128],4,4,4,1,1,1,1,1,1, name='g_f_h5_1', with_w=True)

		h5_1 = tf.nn.relu(tf.contrib.layers.batch_norm(h5_1, scope='g_f_bn5_1'))

		#h5_1, self.h5_w_1, self.h5_wb_1 = deconv3d(h5_1,
		#											[tf.shape(h0)[0],4, 128, 128, 128],4,4,4,1,1,1,1,1,1, name='g_f_h5_2', with_w=True)

		#h5_1 = tf.nn.relu(tf.contrib.layers.batch_norm(h5_1, scope='g_f_bn5_2'))


		h6, self.h6_w, self.h6_wb = deconv3d(h5_1,
													[tf.shape(h0)[0],4, 256, 256, 128],4,4,4,1,2,2,1,1,1, name='g_f_h6', with_w=True)

		h6 = tf.nn.relu(tf.contrib.layers.batch_norm(h6, scope='g_f_bn6'))

		h6_1, self.h6_w, self.h6_wb = deconv3d(h6,
													[tf.shape(h0)[0],4, 256, 256, 3],4,4,4,1,1,1,1,1,1, name='g_f_h6_1', with_w=True)

		h6_1 = tf.nn.relu(tf.contrib.layers.batch_norm(h6_1, scope='g_f_bn6_1'))
	   
		#h6, self.h4_w, self.h6_b = deconv3d(h5,[self.batch_size, 4, 256, 256, 3],1,5,5,1,2,2,1,1,1, name='g_f_h6', with_w=True)
		
		  
		#h6 = tf.nn.relu(tf.contrib.layers.batch_norm(h6, scope='g_f_bn6'))

		mask1 = deconv3d(h6,
								 [tf.shape(h0)[0], 4, 256, 256, 1],5,5,5,1,1,1,1,1,1, name='g_mask1', with_w=False)




		mask = l1Penalty(tf.nn.sigmoid(mask1))

		# mask*h4 + (1 - mask)*

		return tf.nn.tanh(h6_1), mask






def abs_criterion(in_, target):
	return tf.reduce_mean(tf.abs(in_ - target))

def mse_criterion(in_, target):
	return tf.nn.l2_loss(tf.contrib.layers.flatten(in_ - target))
	#return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
