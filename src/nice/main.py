import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds

from nice_model import AdditiveLayer, NICEModel, GaussianLoss

import struct
from array import array
from os.path  import join
import datetime

# Parameters
dataset = "mnist"
num_epochs = 10
save_epoch = 1
layer_type = "additive" # could also be multiplicative
model_path = None

# Training
def train():
    if dataset == 'mnist':
        input_dim = 28*28

    with tf.device(device_name):
      mnist_dataloader = MnistLoader(data_path)
      (im_train, la_train),(im_test, la_test) = mnist_dataloader.load_data()
      print("Dataset loaded")
  
      model = NICEModel(input_dim)
      gauss_loss = GaussianLoss()
      
      # Load model
      if model_path is not None:
          model.load_weights(model_path)
          print("Loaded weights from: {0}".format(model_path))

      opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.01, epsilon=0.001)
      def my_loss(f, diag):
        return gauss_loss.forward(f, diag, input_dim)


      for t in range(num_epochs):
        print("Epoch #{0} ".format(t))
        for input, y in zip(im_train, la_train):
          with tf.GradientTape() as tape:  
            x = tf.Variable(input, dtype=tf.float32)
            tape.watch(x)
            loss = my_loss(model(x), model.scaling_matrix)
            
          print(tf.compat.v1.trainable_variables())
          grads = tape.gradient(loss, x)
          opt.apply_gradients(zip([grads], [x]))
          
        # Save model
        if t % save_epoch == 0:
            fn = "/nice.{0}.e_{1}.pt".format(dataset, t)
            model.save_weights(dir_path + fn)
            print("Saved file: {0}".format(fn))
          
        # Perform validation
        validate(model, my_loss)

    return model

def validate(model, my_loss):
    validation_losses = []
    with tf.device(device_name):  
      for input, _ in zip(im_test, la_test):
        x = tf.Variable(input, dtype=tf.float32)
        validation_losses.append(my_loss(model(x), model.scaling_matrix).numpy())

    print("Validation Loss Statistics: min={0}, med={1}, mean={2}, max={3}".format(np.amin(validation_losses),
                                                                                   np.median(validation_losses),
                                                                                   np.mean(validation_losses),
                                                                                   np.amax(validation_losses)))
    return


if __name__ == '__main__':
    model = train()

    # Generating a couple of samples
    sample_num = 80
    f, axs = plt.subplots(nrows=sample_num//5, ncols=5, figsize=(20, 20))

    for i in range(sample_num):
        gen = model.sample()
        gen = tf.math.sigmoid(gen).cpu()
        
        axs[i//5][i%5].imshow(tf.reshape(gen, [28, 28]), cmap='gray')

    for ax in axs:
        for a in ax:
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_aspect('equal')

    f.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    plt.close('all')
