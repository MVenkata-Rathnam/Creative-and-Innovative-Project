from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import pickle
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.choice = 1
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.text_embedding_dim = 128
    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d1_bn1 = batch_norm(name='d1_bn1')
    self.d1_bn2 = batch_norm(name='d1_bn2')
    self.d2_bn1 = batch_norm(name='d2_bn1')
    self.d2_bn2 = batch_norm(name='d2_bn2')
    if not self.y_dim:
      self.d1_bn3 = batch_norm(name='d1_bn3')
      self.d2_bn3 = batch_norm(name='d2_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    if(self.dataset_name == 'birds'):
	pickle_in_1 = open("./Dataset/birds/train/76images.pickle","rb")
    	self.data = pickle.load(pickle_in_1)
	pickle_in_2 = open("./Dataset/birds/train/char-CNN-RNN-embeddings-Reduced.pickle","rb")
	self.text_embedding = np.array(pickle.load(pickle_in_2))
	shape = self.text_embedding.shape
	self.text_embedding = np.reshape(self.text_embedding,(shape[0],shape[1],shape[2]))
    else:
	pickle_in_1 = open("./Dataset/flowers/train/76images.pickle","rb")
	self.data = pickle.load(pickle_in_1)
	pickle_in_2 = open("./Dataset/flowers/train/char-CNN-RNN-embeddings-Reduced.pickle","rb")
	self.text_embedding = np.array(pickle.load(pickle_in_2))
	shape = self.text_embedding.shape
	self.text_embedding = np.reshape(self.text_embedding,(shape[0],shape[1],shape[2]))
    
    if len(self.data[0].shape) >= 3: #check if image is a non-grayscale image by checking channel number
        self.c_dim = self.data[0].shape[-1]
    else:
	self.c_dim = 1

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def sigmoid_cross_entropy_with_logits(self, x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

  def build_model(self):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim+self.text_embedding_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.mod_z = tf.placeholder(tf.float32,[None,4,4,self.text_embedding_dim], name='mod_z')

    self.G                  = self.generator(self.z, self.y)
    self.D1, self.D1_logits   = self.discriminator_1(inputs, self.y,self.mod_z, reuse=False)
    self.sampler            = self.sampler(self.z, self.y)
    self.D1_, self.D1_logits_ = self.discriminator_1(self.G, self.y,self.mod_z, reuse=True)
    
    self.D2, self.D2_logits = self.discriminator_2(inputs, self.y, self.mod_z, reuse=False)
    self.D2_, self.D2_logits_ = self.discriminator_2(self.G, self.y, self.mod_z, reuse=True)

    self.d1_sum = histogram_summary("d1", self.D1)
    self.d1__sum = histogram_summary("d1_", self.D1_)

    self.d2_sum = histogram_summary("d2", self.D2)
    self.d2__sum = histogram_summary("d2_",self.D2_)
    self.G_sum = image_summary("G", self.G)

    self.d1_loss_real = tf.reduce_mean(
      self.sigmoid_cross_entropy_with_logits(self.D1_logits, tf.ones_like(self.D1)))
    self.d1_loss_fake = tf.reduce_mean(
      self.sigmoid_cross_entropy_with_logits(self.D1_logits_, tf.zeros_like(self.D1_)))

    self.d2_loss_real = tf.reduce_mean(
      self.sigmoid_cross_entropy_with_logits(self.D2_logits, tf.ones_like(self.D2)))
    self.d2_loss_fake = tf.reduce_mean(
      self.sigmoid_cross_entropy_with_logits(self.D2_logits_, tf.zeros_like(self.D2_)))

    if(self.choice == 1):
    	self.g_loss = tf.reduce_mean(
      		self.sigmoid_cross_entropy_with_logits(self.D1_logits_, tf.ones_like(self.D1_)))
    else:
	self.g_loss = tf.reduce_mean(self.sigmoid_cross_entropy_with_logits(self.D2_logits_, 			tf.ones_like(self.D2_)))

    self.d1_loss_real_sum = scalar_summary("d1_loss_real", self.d1_loss_real)
    self.d1_loss_fake_sum = scalar_summary("d1_loss_fake", self.d1_loss_fake)
                          
    self.d1_loss = self.d1_loss_real + self.d1_loss_fake

    self.d2_loss_real_sum = scalar_summary("d2_loss_real", self.d2_loss_real)
    self.d2_loss_fake_sum = scalar_summary("d2_loss_fake", self.d2_loss_fake)

    self.d2_loss = self.d2_loss_real + self.d2_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d1_loss_sum = scalar_summary("d1_loss", self.d1_loss)
    self.d2_loss_sum = scalar_summary("d2_loss", self.d2_loss)

    t_vars = tf.trainable_variables()

    self.d1_vars = [var for var in t_vars if 'd1_' in var.name]
    self.d2_vars = [var for var in t_vars if 'd2_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    d1_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d1_loss, var_list=self.d1_vars)
    d2_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d2_loss, var_list=self.d2_vars)

    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    if(self.choice == 1):
    	self.g_sum = merge_summary([self.z_sum, self.d1__sum, self.G_sum, self.d1_loss_fake_sum, self.g_loss_sum])
    else:
	self.g_sum = merge_summary([self.z_sum, self.d2__sum, self.G_sum, self.d2_loss_fake_sum, self.g_loss_sum])


    self.d1_sum = merge_summary(
        [self.z_sum, self.d1_sum, self.d1_loss_real_sum, self.d1_loss_sum])
    self.d2_sum = merge_summary(
	[self.z_sum, self.d2_sum, self.d2_loss_real_sum, self.d2_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    text_embedding = load_text_embedding(config,self.text_embedding,0)
    sample_z = np.concatenate((sample_z,text_embedding),axis=1)
    sample_mod_z = reshape(text_embedding,4,4).astype(np.float32)
    
    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]
    else:
      sample_files = self.data[0:self.sample_num]
      sample = [
          get_img(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:      
        #self.data = glob(os.path.join(
         # "./Dataset", config.dataset, self.input_fname_pattern))
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        if config.dataset == 'mnist':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_img(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)
	text_embedding = load_text_embedding(config,self.text_embedding,idx)
	batch_z = np.concatenate((batch_z,text_embedding),axis=1)

	batch_mod_z = reshape(text_embedding,4,4).astype(np.float32)

        if config.dataset == 'mnist':
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ 
              self.inputs: batch_images,
              self.z: batch_z,
	      self.mod_z: batch_mod_z,
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z, self.y:batch_labels })
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z, 
              self.y:batch_labels
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels
          })
        else:

          # Update D1 network
          _, summary_str = self.sess.run([d1_optim, self.d1_sum],
            feed_dict={ self.inputs: batch_images, self.z: batch_z, self.mod_z: batch_mod_z })
          self.writer.add_summary(summary_str, counter)

	  # Update D2 network
	  _, summary_str = self.sess.run([d2_optim, self.d2_sum],
	    feed_dict={ self.inputs: batch_images, self.z: batch_z, self.mod_z: batch_mod_z })
	  self.writer.add_summary(summary_str, counter)
          
          errD1_fake = self.d1_loss_fake.eval({ self.z: batch_z, self.mod_z: batch_mod_z })
          errD1_real = self.d1_loss_real.eval({ self.inputs: batch_images, self.mod_z: batch_mod_z })

	  errD2_fake = self.d2_loss_fake.eval({ self.z: batch_z, self.mod_z: batch_mod_z })
	  errD2_real = self.d2_loss_real.eval({ self.inputs: batch_images, self.mod_z: batch_mod_z })

	  # Update condition
	  if(errD1_fake < errD2_fake):
		self.choice = 1
	  else:
		self.choice = 2
	  self.writer.add_summary(summary_str, counter)

	  if(self.choice == 1):
    		self.g_loss = tf.reduce_mean(self.sigmoid_cross_entropy_with_logits(self.D1_logits_, tf.ones_like(self.D1_)))
    	  else:
		self.g_loss = tf.reduce_mean(self.sigmoid_cross_entropy_with_logits(self.D2_logits_, tf.ones_like(self.D2_)))

	  # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z, self.mod_z:batch_mod_z })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          #_, summary_str = self.sess.run([g_optim, self.g_sum],
          #  feed_dict={ self.z: batch_z, self.mod_z: batch_mod_z })
          #self.writer.add_summary(summary_str, counter)

          errG = self.g_loss.eval({self.z: batch_z, self.mod_z: batch_mod_z})
        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d1_loss: %.8f, d2_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD1_fake+errD1_real, errD2_fake+errD2_real, errG))

        if np.mod(counter, 100) == 1:
          if config.dataset == 'mnist':
            samples, d1_loss, d2_loss, g_loss = self.sess.run(
              [self.sampler, self.d1_loss, self.d2_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
              }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d1_loss: %.8f, d2_loss: %.8f, g_loss: %.8f" % (d1_loss, d2_loss, g_loss)) 
          else:
            try:
              samples, d1_loss, d2_loss, g_loss = self.sess.run(
                [self.sampler, self.d1_loss, self.d2_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
		    self.mod_z: sample_mod_z,
                    self.inputs: sample_inputs,
                },
              )
              save_images(samples, image_manifold_size(samples.shape[0]),
                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
              print("[Sample] d1_loss: %.8f, d2_loss: %.8f, g_loss: %.8f" % (d1_loss, d2_loss, g_loss)) 
            except:
              print("one pic error!...")

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator_1(self, image, y=None, z=None, reuse=False):
    with tf.variable_scope("discriminator_1") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        h0 = lrelu(conv2d(image, self.df_dim, name='d1_h0_conv'))
        h1 = lrelu(self.d1_bn1(conv2d(h0, self.df_dim*2, name='d1_h1_conv')))
        h2 = lrelu(self.d1_bn2(conv2d(h1, self.df_dim*4, name='d1_h2_conv')))
        h3 = lrelu(self.d1_bn3(conv2d(h2, self.df_dim*8, name='d1_h3_conv')))
	h3 = tf.concat([h3,z],3)
	h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd1_h4_lin')	

        return tf.nn.sigmoid(h4), h4
      else:
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d1_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(self.d1_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d1_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])      
        h1 = concat([h1, y], 1)
        
        h2 = lrelu(self.d1_bn2(linear(h1, self.dfc_dim, 'd1_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'd1_h3_lin')
        
        return tf.nn.sigmoid(h3), h3

  def discriminator_2(self, image, y=None, z=None, reuse=False):
    with tf.variable_scope("discriminator_2") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        h0 = lrelu(conv2d(image, self.df_dim, name='d2_h0_conv'))
        h1 = lrelu(self.d2_bn1(conv2d(h0, self.df_dim*2, name='d2_h1_conv')))
        h2 = lrelu(self.d2_bn2(conv2d(h1, self.df_dim*4, name='d2_h2_conv')))
        h3 = lrelu(self.d2_bn3(conv2d(h2, self.df_dim*8, name='d2_h3_conv')))
	h3 = tf.concat([h3,z],3)
	h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd2_h4_lin')	

        return tf.nn.sigmoid(h4), h4
      else:
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d2_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(self.d2_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d2_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])      
        h1 = concat([h1, y], 1)
        
        h2 = lrelu(self.d2_bn2(linear(h1, self.dfc_dim, 'd2_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'd2_h3_lin')
        
        return tf.nn.sigmoid(h3), h3

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(
            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(
            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(
            deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def load_mnist(self):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    return X/255.,y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
