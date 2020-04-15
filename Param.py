import torch

# data path
data_PATH = '../input/patches-2000-100-32-32.npz'
output_PATH = '../output/gen_patches-2000-100-32-32.npz'

# output model path
Path_model_saved = '../output/'

# 
GPU_allowed = True # False-training on CPU -- True-training on GPU
Normalize = False # to normalize input data
visualize = True # to visualize data
Save = True # to save generated data
Print = True # print shapes and types

# Training settings
nb_epochs = 2500 # number of epochs    default: 500-AAE / 5000-GAN / 100-AE / 1000-VAE
patch_size = 32

# determine the device for running data
cuda = True if torch.cuda.is_available() and GPU_allowed else False
device = torch.device("cuda") if torch.cuda.is_available() and GPU_allowed else torch.device("cpu")
Tensor = torch.cuda.FloatTensor if cuda and GPU_allowed else torch.FloatTensor


#-----------------------------
#----- Parameters of GAN -----
#-----------------------------
latent_dim = 100
batch_size = 64
channels = 1
data_shape = (channels, patch_size, patch_size)
nb_patches = 1 # number of patches that we want to used in training

# optimizer parameters of GAN architecture
lr = 0.0002
b1 = 0.5
b2 = 0.999


#-----------------------------
#----- Parameters of AAE -----
#-----------------------------
z_dim = 100         # latent dimension
X_dim = patch_size*patch_size       # input dimension
train_batch_size = 100 #100
N = 1000             # number of hidden state layers

# optimizer parameters of AAE architecture
gen_lr = 0.0001
reg_lr = 0.00005


#-----------------------------
#----- Parameters of AE -----
#-----------------------------
lr_AE = 1e-3
batch_size_AE = 1











