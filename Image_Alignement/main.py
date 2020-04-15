import cv2
import numpy as np

from ECC import eccAlign, ConvertWarp_matrix
from FeatureBased import homography_based, AffineTransform_based, Ransac_based, Print_Result
from FFT_Correlation import translation


image_1_path = '../../varna_20190125_153327_0_900_0000001700.jpg'
image_2_path = '../../varna_20190125_153327_0_900_0000002300.jpg'

if __name__ == '__main__':

    # Read the images to be aligned
    im1 =  cv2.imread(image_1_path);
    im2 =  cv2.imread(image_2_path);

    mode = 'ecc'
    feature_mode = '' 

    # Switch between alignment modes
    if mode == "feature":

        if feature_mode == 'homography':
            aligned, warp_matrix = homography_based(im1, im2)
            cv2.imwrite("output.jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(warp_matrix)

        elif feature_mode == 'affine':
            model = AffineTransform_based(im1, im2)
            Print_Result(model)

        elif feature_mode == 'ransac':
            model = Ransac_based(im1, im2)
            Print_Result(model)


    elif mode == "ecc":
        aligned, warp_matrix = eccAlign(im1, im2)
        cv2.imwrite("output.jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(warp_matrix)
        ConvertWarp_matrix(warp_matrix)


    else:
        warp_matrix = translation(im1, im2)
        print(warp_matrix)




