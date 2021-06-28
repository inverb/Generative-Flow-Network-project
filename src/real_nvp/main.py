import torch
from torch import nn
import numpy as np
from torchvision import datasets, transforms
import itertools
import matplotlib.pyplot as plt

from real_nvp_class import LinearBatchNorm, LinearCouplingLayer, Permutation, SequentialFlow, LinearRNVP
from real_nvp_autoencoder import AutoEncoder

EMBEDDING_DIM = 20 # The dimension of the embeddings
FLOW_N = 9 # Number of affine coupling layers
RNVP_TOPOLOGY = [200] # Size of the hidden layers in each coupling layer
AE_EPOCHS = 10 # Epochs for training the autoencoder
NF_EPOCHS = 20 # Epochs for training the normalizing flow
SEED = 0 # Seed of the random number generator
BATCH_SIZE = 100 # Batch size

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the dataset
train_set = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE)

# We use a binary cross-entropy loss for the reconstruction error
loss_f = nn.BCELoss()

# Build the autoencoder
autoencoder = AutoEncoder()
autoencoder = autoencoder.to(device)

optimizer = torch.optim.Adam(itertools.chain(autoencoder.parameters()),
                             lr=1e-3, weight_decay=1e-5)

# Train the autoencoder
for i in range(AE_EPOCHS):
    print('Epoch #{}'.format(i+1))

    losses = []
    for batch_idx, data in enumerate(train_loader):

        x, _ = data
        x = x.to(device)

        # Run the autoencoder
        _x, emb = autoencoder(x)
        _x = torch.sigmoid(_x)

        # Compute loss
        rec_loss = loss_f(_x, x)

        autoencoder.zero_grad()
        rec_loss.backward()
        optimizer.step()

# We replace the original x with the corresponding embedding from the trained autoencoder
embs = []
for batch_idx, data in enumerate(train_loader):

    with torch.no_grad():
        x, y = data

        x = x.to(device)

        _, emb = autoencoder(x)
        for i in range(len(emb)):
            embs.append((emb[i], y[i]))

embs_loader = torch.utils.data.DataLoader(embs, BATCH_SIZE)

# Train the generator
nf_model = LinearRNVP(input_dim=EMBEDDING_DIM, coupling_topology=RNVP_TOPOLOGY, flow_n=FLOW_N, batch_norm=True,
                      mask_type='odds', conditioning_size=10, use_permutation=True, single_function=True)
nf_model = nf_model.to(device)

optimizer1 = torch.optim.Adam(itertools.chain(nf_model.parameters()), lr=1e-4, weight_decay=1e-5)

nf_model.train()
for i in range(NF_EPOCHS):
    print('Epoch #{}'.format(i+1))

    losses = []
    for batch_idx, data in enumerate(embs_loader):

        emb, y = data
        emb = emb.to(device)
        y = y.to(device)
        y = torch.nn.functional.one_hot(y, 10).to(device).float()
        
        # Get the inverse transformation and the corresponding log determinant of the Jacobian
        u, log_det = nf_model.forward(emb, y=y) 

        # Train via maximum likelihood
        prior_logprob = nf_model.logprob(u)
        log_prob = -torch.mean(prior_logprob.sum(1) + log_det)

        nf_model.zero_grad()

        log_prob.backward()

        optimizer1.step()


# Generating some sample of size sample_n
sample_n = 10
f, axs = plt.subplots(nrows=10, ncols=sample_n, figsize=(20, 20))

nf_model.eval()
with torch.no_grad():
    for j in range(10):

        y = torch.nn.functional.one_hot(torch.tensor([j]*sample_n), 10).to(device).float()
        emb, d = nf_model.sample(sample_n, y=y, return_logdet=True)
        z = autoencoder.decoder(emb)

        d_sorted = d.sort(0)[1].flip(0)
        z = z[d_sorted]
        z = torch.sigmoid(z).cpu()
        
        for i in range(sample_n):
            axs[j][i].imshow(z[i].reshape(28, 28), cmap='gray')

for ax in axs:
    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')

f.subplots_adjust(wspace=0, hspace=0)
plt.show()
plt.close('all')
