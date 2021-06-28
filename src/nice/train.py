import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds

import struct
from array import array
from os.path  import join
import datetime

# Parameters
dataset = "mnist"
num_epochs = 20
input_size = 28 * 28
layer_type = "additive" # could also be multiplicative
model_path = None

# Training
def train(args):
    if dataset == 'mnist':
        input_dim = 28*28

    model = NICEModel(input_dim)
    # if (model_path is not None):
    #     assert(os.path.exists(args.model_path)), "[train] model does not exist at specified location"
    #     model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device_name)                                                                           # GPU optimalization
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.01, epsilon=0.001)
    
    def loss(f):
      # TODO

    for t in range(num_epochs):
        print("* epoch {0}:".format(t))
        for input, y in zip(im_train, la_train):
            opt.zero_grad()
            loss_fn(model(input.to(device_name))).backward()
            opt.step()
