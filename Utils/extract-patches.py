"""
Created on Wed Apr  1 16:25:52 2020

@author: arnau
"""

from PIL import Image
import numpy
import time
import os




IMAGES_DIR = "/home/ali/Desktop/Research/Works_MainTopics/varna_20190125_153327_0_900/img/"
IM_WIDTH = 1280
IM_HEIGTH = 720
PATCH_W = 28
PATCH_P = 10



"""Calculate the number of digits of an integer"""
def NumberOfDigits(Number):
    Count = 0
    while(Number > 0):
        Number = Number//10
        Count = Count + 1

    return Count


"""Generate p^2 random coordinates (well spaced) for p^2 patches"""
def generate_corners(p,image_width=IM_WIDTH,image_heigth=IM_HEIGTH,patch_width=PATCH_W):
    l = image_width-patch_width
    h = image_heigth-patch_width
    dl = int(l/p)
    dh = int(h/p)
    corner_list = []
    for i in range(p):
        for j in range(p):
            x = i*dl + numpy.random.randint(dl)
            y = j*dh + numpy.random.randint(dh)
            corner_list.append((y,x)) #when working with arrays, heigth is first
    return corner_list


"""Extract small patches described by the list l from the image arr"""
def extract_patches(arr,l):
    p = len(l)
    lp = []
    for (y,x) in l:
        lp.append(arr[y:y+PATCH_W,x:x+PATCH_W])
    a = numpy.array(lp,copy=True)
    return a


start_time = time.process_time()
l = generate_corners(PATCH_P)

"""Loading and processing the images"""
im_l = []
nb = '000000'
while True:
    print(nb)

    frame = Image.open(IMAGES_DIR+'varna_20190125_153327_0_900_0000'+nb+'.jpg')

    #arr = numpy.array(frame.convert('RGB'))
    arr = numpy.array(frame.convert('L'))
    im_l.append(extract_patches(arr,l))


    if (int(nb) == 199900):
        break

    nb = int(nb)+100
        
    nb_digits = NumberOfDigits(int(nb))
    #print(nb_digits)

    nb = str('0'*(6-nb_digits)) + str(nb)
    #print(nb)


a = numpy.array(im_l,copy=True)
print(a.shape,a.dtype)
numpy.savez_compressed('../input/patches-2000-100-28-28.npz',a)

print("Processing time = ", time.process_time()-start_time, " s")










