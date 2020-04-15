from PIL import Image

import torch

from Param import patch_size

import numpy as np

# concatenate 2 images side by side -- to save or visulaize the real image with its generated
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

# Vizualizing one sample of an input and its generated
def Visualization(in_image, out_image):
    patch = Image.fromarray(np.uint8(in_image.cpu().numpy()) , 'L')
    gen_patch = Image.fromarray(np.uint8(out_image.cpu().detach().numpy()) , 'L')
    
    dst = get_concat_h(patch, gen_patch)
    dst.show()

# save model of AAE
def save_model(model, filename):
    print('Best model so far, saving it...')
    torch.save(model.state_dict(), filename)  # filename example: conv_autoencoder.pth 

# Print different types of losses of AAE
def report_loss(epoch, D_loss_gauss, G_loss, recon_loss):

    print('Epoch-{}; D_loss_gauss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'.format(epoch,
                                                                          D_loss_gauss.item(),
                                                                          G_loss.item(),
                                                                          recon_loss.item()))

# Normalizing the input data to be in range [0,1]
def Normalization(pp, verbose=True):
    pp_tmp = pp.view(pp.shape[0], pp.shape[1], -1)
    average = torch.mean(pp_tmp, axis = 2)
    sigma = torch.var(pp_tmp,axis = 2)

    if verbose: print('data',pp.shape,pp.dtype)
    if verbose: print('Average',average.shape,average.dtype)
    if verbose: print('Variance',sigma.shape,sigma.dtype)

    Normalized_pp = ((pp_tmp.permute(2,0,1) - average[None,:,:])/sigma[None,:,:]).permute(1,2,0).view(pp.shape[0], pp.shape[1],patch_size,patch_size)
    if verbose: print('Normalized data',Normalized_pp.shape,Normalized_pp.dtype)

    #Normalized_pp = (Normalized_pp - 0.5)/0.5

    return Normalized_pp, average, sigma



"""
def create_latent(Q, loader):
    '''
    Creates the latent representation for the samples in loader
    return:
        z_values: numpy array with the latent representations
        labels: the labels corresponding to the latent representations
    '''
    Q.eval()
    labels = []

    for batch_idx, (X, target) in enumerate(loader):

        X = X * 0.3081 + 0.1307
        # X.resize_(loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        labels.extend(target.data.tolist())
        if cuda:
            X, target = X.cuda(), target.cuda()
        # Reconstruction phase
        z_sample = Q(X)
        if batch_idx > 0:
            z_values = np.concatenate((z_values, np.array(z_sample.data.tolist())))
        else:
            z_values = np.array(z_sample.data.tolist())
    labels = np.array(labels)

    return z_values, labels
"""

