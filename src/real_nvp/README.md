# Real NVP model

Model introduced [here](https://arxiv.org/pdf/1605.08803.pdf) by Dinh et. all. 
As it is more complex than NICE, we didn't write code ourselves, but instead downloaded it from [this](https://github.com/ispamm/realnvp-demo-pytorch) github repository.

## Model overlook

The idea of Real NVP (Real-value Non Volume Preserving) model is very similar to NICE models. They differ only on flow function, as instead of just one additive factor **m**, Real NVP introduces two factors **s, t**, where **s** is muliplicative.

![Real NVP equations](../../docs/real_nvp_equations.png)

Where &bigodot; is element-wise multiplication. As we can see above, introducing multiplication don't make inversing functions any harder, and yet it can vastly improve results.

## Training 

As the model obtains its database from ```tensorflow.databases```, no additional data preparation is neccessary. Standard, 30 epoch long training can be started with

```
python main.py
```

After a very short training of 10 epochs for autoencoder and then 20 epochs for the model, we obtain good results for generating MNIST. Below are examples of generated digits.

![results](../../docs/result.png)
