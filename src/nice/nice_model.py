import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds

import struct
from array import array
from os.path  import join
import datetime

class AdditiveLayer(tf.keras.layers.Layer):
  def __init__(self, input_size, parity):
    super(AdditiveLayer, self).__init__()
    self.input_size = input_size
    self.parity = parity  # if 0, first half is left unchanged

  def build(self, input_shape):
    self.kernel1 = self.add_weight("kernel",
                                   shape=[self.input_size/2,
                                          self.input_size/2])
    self.kernel2 = self.add_weight("kernel",
                                   shape=[self.input_size/2,
                                          self.input_size/2])

  def call(self, inputs):
    if self.parity == 0:
      factors = inputs[:,:self.input_size/2]
    else:
      factors = inputs[:,(self.input_size/2 + 1):]
    mult1 = tf.matmul(inputs, self.kernel1)
    mult2 = tf.matmul(mult1,  self.kernel2)
    
    if self.parity == 0:
      res = tf.concat(tf.zeros([tf.input_size/2]), mult2)
    else:
      res = tf.concat(fun, tf.zeros([tf.input_size/2]))
    return tf.add(inputs, res)

  def inverse(self, inputs):
    if self.parity == 0:
      factors = inputs[:,:self.input_size/2]
    else:
      factors = inputs[:,(self.input_size/2 + 1):]
    mult1 = tf.matmul(inputs, self.kernel1)
    mult2 = tf.matmul(mult1,  self.kernel2)
    
    if self.parity == 0:
      res = tf.concat(tf.zeros([tf.input_size/2]), mult2)
    else:
      res = tf.concat(fun, tf.zeros([tf.input_size/2]))
    return tf.substract(inputs, res)


class NICEModel(tf.keras.Model):
  def __init__(self, input_size):
    super(NICEModel, self).__init__(name='')

    model = []
    for i in range(6):
      layer = AdditiveLayer(input_size, i % 2)
      model.append(layer)

  def call(self, input_tensor, training=False):
    x = input_tensor
    for i in range(6):
      layer = model[i]
      x = layer(x)
    return x
  
  def inverse(input_tensor, training=False):
    x = input_tensor
    for i in range(5,-1,-1):
      layer = model[i]
      x = layer(x)
    return x
