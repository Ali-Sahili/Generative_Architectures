import cv2
import numpy as np

from math import cos, atan

"""
# Enhanced Correlation Coefficient (ECC) Algorithm
  1. Read the images.
  2. Convert them to grayscale.
  3. Pick a motion model you want to estimate.
  4. Allocate space (warp_matrix) to store the motion model.
  5. Define a termination criteria that tells the algorithm when to stop.
  6. Estimate the warp matrix using findTransformECC.
  7. Apply the warp matrix to one of the images to align it with the other image.

# Define the motion model
         cv2.MOTION_TRANSLATION      # shifting without rotation. scale or shear
         cv2.MOTION_EUCLIDEAN        # shifting and rotation without scale or shear
         cv2.MOTION_AFFINE           # any 2D affine transformation
         cv2.MOTION_HOMOGRAPHY       # for almost all 3D effects

"""
# Enhanced Correlation Coefficient (ECC) Maximization
def eccAlign(im1, im2, warp_mode = cv2.MOTION_AFFINE, termination_eps = 1e-8, nb_iterations = 5000):

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, nb_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return im2_aligned, warp_matrix



# for affine warp_matrix 2x3
def ConvertWarp_matrix(warp_matrix):

   tx = warp_matrix[0][2]
   ty = warp_matrix[1][2]

   alpha = warp_matrix[0][0]
   beta = warp_matrix[0][1]

   angle = atan(beta/alpha)
   scale = alpha/cos(angle)

   print("Scale: ", scale)
   print("Translation: ", tx, ty)
   print("Rotation: ", angle)
  # print('with confidence; ', cc*100)
   print()
















