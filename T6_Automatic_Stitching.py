from typing import List, Any

import cv2
from enum import Enum
import numpy as np
import sys
import os


class Methods(Enum):
    SIFT = 1
    ORB = 2

if __name__ == '__main__':


    image_1 = cv2.imread("C:/Users/Christian Forero/Desktop/imagenes/6.jpeg")
    image_gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_draw_1 = np.copy(image_1)
    image_2 = cv2.imread("C:/Users/Christian Forero/Desktop/imagenes/7.jpeg")
    image_gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    image_draw_2 = np.copy(image_2)
    image_3 = cv2.imread("C:/Users/Christian Forero/Desktop/imagenes/8.jpeg")
    image_gray_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
    image_draw_3 = np.copy(image_3)

    # sift/orb interest points and descriptors
    method = Methods.SIFT
    if method == Methods.SIFT:
        sift = cv2.SIFT_create(nfeatures=50)   # shift invariant feature transform
        keypoints_1, descriptors_1 = sift.detectAndCompute(image_gray_1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(image_gray_2, None)
        keypoints_3, descriptors_3 = sift.detectAndCompute(image_gray_3, None)
    else:
        orb = cv2.ORB_create(nfeatures=50)     # oriented FAST and Rotated BRIEF
        keypoints_1, descriptors_1 = orb.detectAndCompute(image_gray_1, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(image_gray_2, None)
        keypoints_3, descriptors_3 = orb.detectAndCompute(image_gray_3, None)

    image_draw_1 = cv2.drawKeypoints(image_gray_1, keypoints_1, None)
    image_draw_2 = cv2.drawKeypoints(image_gray_2, keypoints_2, None)
    image_draw_3 = cv2.drawKeypoints(image_gray_3, keypoints_3, None)

    # Interest points matching
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=1)
    matches_1 = bf.knnMatch(descriptors_2, descriptors_3, k=1)

    image_matching = cv2.drawMatchesKnn(image_1, keypoints_1, image_2, keypoints_2, matches, None)
    image_matching_1 = cv2.drawMatchesKnn(image_2, keypoints_2, image_3, keypoints_3, matches_1, None)

    # Retrieve matched points
    points_1 = []
    points_2 = []
    points_3 = []
    for idx, match in enumerate(matches):
        idx2 = match[0].trainIdx
        points_1.append(keypoints_1[idx].pt)
        points_2.append(keypoints_2[idx2].pt)

    for idx, match in enumerate(matches_1):
         idx3 = match[0].trainIdx
         points_2.append(keypoints_2[idx2].pt)
         points_3.append(keypoints_3[idx3].pt)

    print(len(points_1))
    points_2=points_2[0:len(points_1)]
    print(len(points_2))

    # # Compute homography and warp image_1
    H, status = cv2.findHomography(np.array(points_1), np.array(points_2), method=cv2.RANSAC)
    image_warped_12 = cv2.warpPerspective(image_1, H, (image_1.shape[1], image_1.shape[0]))

    points_3=points_3[0:len(points_2)]
    print(len(points_3))

    H2, status = cv2.findHomography(np.array(points_2), np.array(points_3), method=cv2.RANSAC)
    image_warped_23 = cv2.warpPerspective(image_2, H2, (image_2.shape[1], image_2.shape[0]))

    # cv2.imshow("Image 1", image_1)
    # cv2.imshow("Image 2", image_2)
    # cv2.imshow("Image 3", image_3)

    cv2.imshow("Image matching", image_matching)
    cv2.imshow("Image warped", image_warped_12)
    #cv2.imshow("Image warped-1", image_warped_23)
    cv2.waitKey(0)
