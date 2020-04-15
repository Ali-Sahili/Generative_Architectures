import cv2
import numpy as np

from skimage.transform import AffineTransform
from skimage.measure import ransac


# (ORB,SIFT or SURF) feature based alignment      
def featureAlign(im1, im2, detector = 'SIFT', max_features = 5000, feature_retention = 0.15, MIN_MATCH_COUNT = 10):
  
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    
    if detector == 'ORB':
        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(max_features)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)
  
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * feature_retention)
        matches = matches[:numGoodMatches]

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

  
    else:
        if detector == 'SIFT':
            # Detect SIFT features and compute descriptors.
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
            keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)

        elif detector == 'SURF':
            # Detect SIFT features and compute descriptors.
            surf = cv2.xfeatures2d.SURF_create()
            keypoints1, descriptors1 = surf.detectAndCompute(im1Gray, None)
            keypoints2, descriptors2 = surf.detectAndCompute(im2Gray, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors1,descriptors2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            points1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,2)
            points2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,2)

    return points1, points2



    """ First approach """
def homography_based(im1, im2, detector = 'SIFT', max_features = 5000, feature_retention = 0.15, MIN_MATCH_COUNT = 10):

    points1, points2 = featureAlign(im1, im2, detector, max_features, feature_retention, MIN_MATCH_COUNT)    

    # Find homography
    homography_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, homography_matrix, (width, height))

    return im1Reg, homography_matrix


    """ Second approach """
def AffineTransform_based(im1, im2, detector = 'SIFT', max_features = 5000, feature_retention = 0.15, MIN_MATCH_COUNT = 10):

    points1, points2 = featureAlign(im1, im2, detector, max_features, feature_retention, MIN_MATCH_COUNT)     

    # estimate affine transform model using all coordinates
    model = AffineTransform()
    model.estimate(points1, points2)

    return model


    """ Third approach """
def Ransac_based(im1, im2, detector = 'SIFT', max_features = 5000, feature_retention = 0.15, MIN_MATCH_COUNT = 10):

    points1, points2 = featureAlign(im1, im2, detector, max_features, feature_retention, MIN_MATCH_COUNT)    

    # robustly estimate affine transform model with RANSAC
    model_robust, inliers = ransac((points1, points2), AffineTransform, min_samples=3,
                               residual_threshold=2, max_trials=100)
    outliers = inliers == False

    return model_robust 

# Print the parameters of the transformation model
def Print_Result(model):

    print("Scale: ", model.scale[0], model.scale[1])
    print("Translation: ", model.translation[0], model.translation[1])
    print("Rotation: ", model.rotation)
    print()



















