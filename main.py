from PIL import Image

import torch

import numpy as np

from Param import nb_epochs, nb_patches, latent_dim, patch_size, device
from Param import data_PATH, output_PATH, Normalize, visualize, Save, Print
from Param import train_batch_size, Path_model_saved

from Param import batch_size_AE

from train.AE import fit_AE
from train.VAE import fit_VAE
from train.GAN import generate_model
from train.AAE import generate_model_AAE

from Utils.utils import Visualization, save_model, Normalization

import time


# helping message
print("Available methods are:")
print("   1. AE")
print("   2. VAE")
print("   3. GAN")
print("   4. AAE")
print()


# input from user
METHOD = input("Choose method: ")

# import data
extension = data_PATH.split('.')[-1]
if extension == 'npz':
    pp1 = torch.from_numpy(np.load(data_PATH)['arr_0']).type(torch.float).to(device)

elif extension == 'npy':
    pp1 = torch.from_numpy(np.load(data_PATH)).type(torch.float).to(device)

else:
    assert False, "Undefined type of the data file! Please use data with type npz or npy."


# take only 2000 from 9000
pp = pp1[0:2000]
denorm_pp = pp


# Normalize input data
if Normalize:
    pp, average_forNorm, sigma_forNorm = Normalization(pp, verbose=True)

#---------------------------------
#------------- AE ----------------
#---------------------------------
if METHOD == 'AE':
    #
    data = pp[:,0,:,:].view(2000,patch_size * patch_size)
    if Print: print('input data: ',data.shape)

    train_data = data.view(100,20,patch_size * patch_size)

    # Starting Training
    start_time = time.process_time()
    model = fit_AE(train_data)

    print("Processing time = ", time.process_time()-start_time, " s")
    print("Finishing ...")

    data_test = pp[:,1,:,:].view(2000,patch_size * patch_size)
    average = torch.mean(data_test[0])
    sigma = torch.var(data_test[0])

    print('average: ', average)
    print('sigma: ', sigma)
    out, aver = model(data_test[0])
    print('output average: ', aver)
    print('output: ', out.shape)

    if visualize:
        Visualization(data_test[0].view(patch_size, patch_size), out.view(patch_size, patch_size)) 

    #-------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------

#---------------------------------
#------------- VAE ----------------
#---------------------------------
elif METHOD == 'VAE':
    #
    data = pp[:,0,:,:].view(2000,patch_size * patch_size)
    if Print: print('input data: ',data.shape)

    train_data = data.view(100,20,patch_size * patch_size)

    # Starting Training
    start_time = time.process_time()
    model = fit_VAE(train_data)

    print("Processing time = ", time.process_time()-start_time, " s")
    print("Finishing ...")

    data_test = pp[:,1,:,:].view(2000,patch_size * patch_size)
    average = torch.mean(data_test)
    sigma = torch.var(data_test)

    print('average: ', average)
    print('sigma: ', sigma)
    out, mu, logvar = model(data_test)
    print('output average: ', torch.mean(mu))
    print('output var: ', torch.mean(logvar))
    print('output: ', out.shape)


    if visualize:
        Visualization(data_test[0].view(patch_size, patch_size), out[0].view(patch_size, patch_size)) 

    #-------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------




#---------------------------------
#------------- AAE ---------------
#---------------------------------
elif METHOD == 'AAE':
    #
    data = pp[:,0,:,:].view(2000,patch_size*patch_size)
    if Print: print('input data: ',data.shape)

    # prepare data as a list of tensor of batch_size x 1024 -- 1024 = 32 x 32
    train_data = []
    for i in range(0,data.shape[0],train_batch_size):
        train_data.append(data[i:i+train_batch_size])

    if Print: print('train data: ',len(train_data), len(train_data[0]), len(train_data[0][0]))

    # Starting Training
    start_time = time.process_time()
    Q, P, gen_data = generate_model_AAE(train_data)

    print("Processing time = ", time.process_time()-start_time, " s")
    print("Finishing ...")

    if Save:
        filename1 = 'AAE_model_Encoder.pth'
        filename2 = 'AAE_model_EncoderDecoder.pth'
        save_model(Q, Path_model_saved + filename1)
        save_model(P, Path_model_saved + filename2)

    gen_data = gen_data.view(train_batch_size, patch_size, patch_size)
    if Print: print('generated data: ',gen_data.shape)


    """
    X = train_data[10]
    aa = Q(X)
    print(aa)
    print(aa.shape)
    """

    # Visualize an example of generated data and its real correspondance
    if visualize:
        #gen_data = gen_data*0.5 + 0.5
        generated_data = gen_data[0]*sigma[0,0] + average[0,0]

        Visualization(denorm_pp[0,0], generated_data)

    #-----------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------




#---------------------------------
#------------- GAN ---------------
#---------------------------------
elif METHOD == 'GAN':

    # Starting Training
    start_time = time.process_time()
    gen_data = generate_model(pp, latent_dim, nb_epochs, nb_patches)

    if Print: print('real data:',pp[:,:nb_patches].shape,pp.dtype)
    if Print: print('generated data:',gen_data.shape,gen_data.dtype)

    print("Processing time = ", time.process_time()-start_time, " s")

    print("Finishing ...")

    # Denormalize generated data if the input data was normalized
    if Normalize: 
        #gen_data = gen_data*0.5 + 0.5
        gen_data_tmp = gen_data.view(gen_data.shape[0], gen_data.shape[1], -1)
        denorm_gen_data = (gen_data_tmp.permute(2,0,1)*sigma[None,:,:nb_patches] + average[None,:,:nb_patches]).permute(1,2,0).view(gen_data.shape[0], gen_data.shape[1],patch_size,patch_size)

        if Print: print('Denormalized generated data:',denorm_gen_data.shape,denorm_gen_data.dtype)

        gen_data = denorm_gen_data

    # Saving generated data 
    if Save:
        np.savez_compressed(output_PATH,np.uint8(gen_data.detach().numpy()))


    # Visualize an example of generated data and its real correspondance
    if visualize:
        Visualization(denorm_pp[0,0], gen_data[0,0])

    #--------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------


else:
    assert False, "Undefined used method !!!"

















