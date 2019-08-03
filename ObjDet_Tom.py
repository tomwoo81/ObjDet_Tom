# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:51:13 2019

@author: Tom
"""

import numpy as np
import cv2 as cv

def ObjDet_Tom(filename1, filename2):
    STEP = (5, 5)
    COUNT_OF_BEST_MATCHES_PER_DESC = 2
    LOWE_RATIO = 0.5
    MIN_GOOD_MATCH_RATIO = 0.1
    DISTANCE_COEFFICIENT = 0.5
    SEARCH_COEFFICIENT = 0.6
    
    original = cv.imread(filename1)
    cv.imshow('Original', original)
    print("Shape of original image: {}".format(original.shape))
    
    target = cv.imread(filename2)
    cv.imshow('Target', target)
    print("Shape of target image: {}".format(target.shape))
    
    originalGray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    
    targetGray = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
    
    h_o, w_o = original.shape[:2]
    h_t, w_t = target.shape[:2]
    
    if w_t > w_o:
        print("Width of the small image ({}) is larger than that of the large image ({})!".format(w_t, w_o))
        return []
    
    if h_t > h_o:
        print("Height of the small image ({}) is larger than that of the large image ({})!".format(h_t, h_o))
        return []
    
    sift = cv.xfeatures2d.SIFT_create()
    
    cntkeypointsT = 0
    keypointsTList = []
    descTList = []
    
    for c in range(3):
        keypoints, desc = sift.detectAndCompute(target[..., c], None)
        print("Count of keypoints[{:d}]: {:d}".format(c, len(keypoints)))
        print("Shape of descriptors[{:d}]: {}".format(c, desc.shape))
        cntkeypointsT += len(keypoints)
        keypointsTList.append(keypoints)
        descTList.append(desc)
    print("Total count of keypoints: {:d}".format(cntkeypointsT))
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    searchParams = dict(checks = 50)
    matcher = cv.FlannBasedMatcher(indexParams, searchParams)
    
    # Resize for an image pyramid (optional)
    
    original1 = original.copy()
    candRegionList = []
    
    for b in range(h_t - 1, h_o, STEP[0]):
        for r in range(w_t - 1, w_o, STEP[1]):
            window = original[b+1-h_t:b+1, r+1-w_t:r+1]
            goodMatches = []
            winFlag = True
            for c in range(3):
                keypoints, desc = sift.detectAndCompute(window[..., c], None)
                colflag = False
                if len(keypoints) >= COUNT_OF_BEST_MATCHES_PER_DESC:
                    matches = matcher.knnMatch(descTList[c], desc, k = COUNT_OF_BEST_MATCHES_PER_DESC)
                    for m, n in matches:
                        if m.distance < LOWE_RATIO * n.distance:
                            goodMatches.append(m)
                            colflag = True
                if not colflag:
                    winFlag = False
                    break
            if not winFlag:
                continue
            if len(goodMatches) >= MIN_GOOD_MATCH_RATIO * cntkeypointsT:
                cv.rectangle(original1, (r - w_t + 1, b - h_t + 1), (r, b), (0, 0, 255))
                candRegionList.append((b - h_t + 1, r - w_t + 1, b, r))
    cv.imshow('Candidate regions - 1', original1)
    print("Count of Candidate regions - 1: {:d}".format(len(candRegionList)))
    
    original2 = original.copy()
    candRegionList2 = []
    
    for i in range(len(candRegionList)):
        discardFlag = False
        for j in range(i + 1, len(candRegionList)):
            dist_h = abs(candRegionList[i][1] - candRegionList[j][1])
            dist_v = abs(candRegionList[i][0] - candRegionList[j][0])
            if (dist_h < DISTANCE_COEFFICIENT * w_t) and (dist_v < DISTANCE_COEFFICIENT * h_t):
                discardFlag = True
                break
        if not discardFlag:
            cv.rectangle(original2, (candRegionList[i][1], candRegionList[i][0]), 
                         (candRegionList[i][3], candRegionList[i][2]), (0, 0, 255))
            candRegionList2.append(candRegionList[i])
    cv.imshow('Candidate regions - 2', original2)
    print("Count of Candidate regions - 2: {:d}".format(len(candRegionList2)))
    
    original3 = original.copy()
    detectedObjectList = []
    
    for candRegion in candRegionList2:
        croppedGrayTop = int(candRegion[0]-SEARCH_COEFFICIENT*h_t)
        if croppedGrayTop < 0:
            croppedGrayTop = 0
        croppedGrayLeft = int(candRegion[1]-SEARCH_COEFFICIENT*w_t)
        if croppedGrayLeft < 0:
            croppedGrayLeft = 0
        croppedGrayBottom = int(candRegion[2]+SEARCH_COEFFICIENT*h_t)
        if croppedGrayBottom >= h_o:
            croppedGrayBottom = h_o - 1
        croppedGrayRight = int(candRegion[3]+SEARCH_COEFFICIENT*w_t)
        if croppedGrayRight >= w_o:
            croppedGrayRight = w_o - 1
        w_c = croppedGrayRight - croppedGrayLeft + 1
        h_c = croppedGrayBottom - croppedGrayTop + 1
        croppedGray = originalGray[croppedGrayTop:croppedGrayBottom+1, croppedGrayLeft:croppedGrayRight+1]
        minEuclideanDist = np.inf
        for b in range(h_t - 1, h_c):
            for r in range(w_t - 1, w_c):
                window = croppedGray[b+1-h_t:b+1, r+1-w_t:r+1]
                euclideanDist = np.sqrt(((window - targetGray) ** 2).sum())
                if euclideanDist < minEuclideanDist:
                    minEuclideanDist = euclideanDist
                    matchedWindow = (b, r)
        detectedObjectList.append((croppedGrayTop + matchedWindow[0] - h_t + 1, croppedGrayLeft + matchedWindow[1] - w_t + 1, 
                                   croppedGrayTop + matchedWindow[0], croppedGrayLeft + matchedWindow[1]))
    for detectedObject in detectedObjectList:
        cv.rectangle(original3, (detectedObject[1], detectedObject[0]), 
                     (detectedObject[3], detectedObject[2]), (0, 0, 255))
    cv.imshow('Detected objects', original3)
    print("Count of detected objects: {:d}".format(len(detectedObjectList)))
    
    return detectedObjectList

if __name__ is "__main__":
    import sys
    import time
    
    filename1, filename2 = sys.argv[1], sys.argv[2]
    
    # Time the execution of the function
    start = time.time()
    retList = ObjDet_Tom(filename1, filename2)
    end = time.time()
    
    print("-" * 30)
    print("Time elapsed: {:.3f} s".format(end - start))
    print("Count of detected objects: {:d}".format(len(retList)))
    print("Regions of detected objects: ")
    for ret in retList:
        print(ret)

# End of file
