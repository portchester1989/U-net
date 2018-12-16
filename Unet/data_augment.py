import pandas as pd
import numpy as np
import scipy as sc
import elasticdeform
import cv2
import imgaug.augmenters as iaa
import imgaug.imgaug as ia
def elastic_bbox(X,bbox,displacement,order=3,mode='constant', cval=0.0, crop=None, prefilter=True,bbox_mode = 'rotated'):
    '''
    if isinstance(X, numpy.ndarray):
        Xs = [X]
    elif isinstance(X, list):
        Xs = X
    else:
        raise Exception('X should be a numpy.ndarray or a list of numpy.ndarrays.')
    assert len(Xs) > 0, 'You must provide at least one image.'
    assert all(isinstance(x, numpy.ndarray) for x in Xs), 'All elements of X should be numpy.ndarrays.'
    '''
    #create a mask and apply transformation on the mask
    mask = np.zeros(X.shape[:2],dtype = 'uint8')
    (x,y,w,h) = bbox
    cv2.rectangle(mask,(x,y),(x+w,y+h),255,-1)
    mask = elasticdeform.deform_grid(mask,displacement,axis = (0,1))
    #calculate mean intensity outside of mask
    constant_intensity = np.mean(X[mask < 127,:],axis = (0,1))
    ret,thresh = cv2.threshold(mask,127,255,0)
    contours = cv2.findContours(thresh, mode=1,method=2)
    cnt = contours[0]
    if bbox_mode == 'rotated':
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        # four courners 
        #box = np.int0(box)
        return box
    else:
        x,y,w,h = cv2.boundingRect(cnt)
        #width and height
        return (x,y,w,h),constant_intensity
def random_elastic_bbox(X,bboxes,sigma=25,points=3,mode='constant',cval=0.0,crop=None,prefilter=True,bbox_mode='rotated'):
    #random displacement
    if not isinstance(bboxes,list):
        bboxes = [bboxes]
    if not isinstance(points,(list,tuple)):
        points = [points] * 2
    displacement = np.random.randn(2,*points) * sigma
    displaced_image = elasticdeform.deform_grid(X, displacement, order = 3, mode = 'constant', cval = 0.0, crop=None, prefilter = True,axis=(0,1))
    displaced_bbox = [elastic_bbox(X,bbox,displacement,bbox_mode=bbox_mode) for bbox in bboxes]
    return (displaced_image,displaced_bbox)
def boxes_within_image(X,bboxes_list,aug,elastic = True):
    if not isinstance(X,list):
        X = [X]
        bboxes_list = [bboxes_list]
    if elastic == True:
        pairs = [random_elastic_bbox(image,bboxes,bbox_mode='upright') for image,bboxes in zip(X,bboxes_list)]
    else:
        pairs = [(image,bboxes) for image,bboxes in zip(X,bboxes_list)]
    #convert bboxes to imgaug objects
    bboxes = ia.BoundingBoxesOnImage([ia.BoundingBox(x1 = pair[1][0][0], y1 = pair[1][0][1], x2 = pair[1][0][2] + pair[1][0][0],y2 = pair[1][0][1] + pair[1][0][3]) for pair in pairs],shape=X[0].shape)
    image_aug = aug.augment_images([pair[0] for pair in pairs])
    bboxes_aug = aug.augment_bounding_boxes([bboxes])
    #retain bounding boxes within image regions
    bboxes_aug = [[bbox for bbox in bboxes.bounding_boxes if bbox.is_fully_within_image(X[0].shape)] for bboxes in bboxes_aug]
    #return image_aug
    return (image_aug,bboxes_aug)
