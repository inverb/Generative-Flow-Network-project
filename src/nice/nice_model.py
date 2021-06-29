import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds

# Contains model, layer and loss classes


class AdditiveLayer(tf.keras.layers.Layer):
  def __init__(self, input_size, parity):
    super(AdditiveLayer, self).__init__()
    self.input_size = input_size
    self.parity = parity  # if 0, first half is left unchanged

  def build(self, input_shape):
    self.kernel1 = self.add_weight("kernel1",
                                   shape=[self.input_size//2, self.input_size//2],
                                   dtype=tf.float32, trainable=True)
    self.kernel2 = self.add_weight("kernel2",
                                   shape=[self.input_size//2, self.input_size//2],
                                   dtype=tf.float32, trainable=True)

  def call(self, inputs):
    if self.parity == 0:
      factors = inputs[:,:self.input_size//2]
    else:
      factors = inputs[:,(self.input_size//2):]
    mult1 = tf.matmul(factors, self.kernel1)
    mult1 = tf.keras.activations.relu(mult1)
    mult2 = tf.matmul(mult1,  self.kernel2)
    
    if self.parity == 0:
      res = tf.concat([tf.zeros([1, self.input_size//2], dtype=tf.float32), mult2], axis=1)
    else:
      res = tf.concat([mult2, tf.zeros([1, self.input_size//2], dtype=tf.float32)], axis=1)
    return tf.add(inputs, res)

  def inverse(self, inputs):
    if self.parity == 0:
      factors = inputs[:,:self.input_size//2]
    else:
      factors = inputs[:,(self.input_size//2):]
    mult1 = tf.matmul(factors, self.kernel1)
    mult1 = tf.keras.activations.relu(mult1)
    mult2 = tf.matmul(mult1, self.kernel2)
    
    if self.parity == 0:
      res = tf.concat([tf.zeros([1, self.input_size//2], dtype=tf.float32), mult2[0]], axis=1)
    else:
      res = tf.concat([mult2[0], tf.zeros([1, self.input_size//2], dtype=tf.float32)], axis=1)
    return tf.subtract(inputs, res)


class NICEModel(tf.keras.Model):
  def __init__(self, input_size):
    super(NICEModel, self).__init__(name='')

    self.seq_model = []
    for i in range(6):
      layer = AdditiveLayer(input_size, i % 2)
      self.seq_model.append(layer)
    self.input_size = input_size
    self.scaling_matrix = tf.ones(input_size)

  def call(self, input_tensor, training=False):
    x = tf.expand_dims(input_tensor, 0)
    for i in range(6):
      layer = self.seq_model[i]
      x = layer(x)
    x = tf.matmul(x, tf.linalg.tensor_diag(tf.math.exp(self.scaling_matrix)))
    return x
  
  def inverse(self, input_tensor, training=False):
    x = tf.matmul(tf.expand_dims(input_tensor, 0),
                  tf.linalg.tensor_diag(tf.math.exp(self.scaling_matrix)))
    for i in range(5,-1,-1):
      layer = self.seq_model[i]
      x = layer(x)
    return x[0]

  def sample(self):
    x = tf.random.normal([self.input_size])
    z = self.inverse(x)
    return z


# Loss function
class GaussianLoss(tf.keras.losses.Loss):
    def __init__(self):
      super(GaussianLoss, self).__init__()

    def forward(self, h, diag, input_dim):
      # Equation from paper:
      # \sum^D_i s_{ii} - { (1/2) * \sum^D_i  h_i**2) + (D/2) * log(2\pi) } 
      loglkhd = 0.5 * tf.math.reduce_sum(tf.math.pow(h,2)) + input_dim*0.5*tf.math.log(tf.convert_to_tensor(2*np.pi))
      return tf.math.reduce_mean(loglkhd - tf.math.reduce_sum(diag))
