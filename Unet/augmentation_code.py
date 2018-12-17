ugment them and save somewhere
#mean intensity out of the bounding box region
def augment_and_save(images,bboxes,num_aug,save_path,load = True):
    aug = iaa.SomeOf((1,2),[iaa.ElasticTransformation(alpha=(0, 3.0), sigma=0.25,mode='constant',cval = 217),iaa.Affine(rotate = (-45,45),translate_percent = (-.1,.1),mode='constant',cval = 217)])
    bbs = [ia.BoundingBoxesOnImage([ia.BoundingBox(x1 = bboxes[j,0],y1 = bboxes[j,1], x2 = bboxes[j,0] + bboxes[j,3],y2 = bboxes[j,1] + bboxes[j,2])],shape=(256,256)) for j in range(len(bboxes))]
    aug_boxes_list = []
    aug_images_list = []
    for i in range(num_aug):
        aug_images = np.asarray(aug.augment_images(images))
        aug_bbs = aug.augment_bounding_boxes(bbs)        
        within_image_indices = [k for k in range(len(bboxes)) if aug_bbs[k].bounding_boxes[0].is_fully_within_image((256,256))]                          aug_boxes = np.asarray([(aug_bbs[k].bounding_boxes[0].x1_int,aug_bbs[k].bounding_boxes[0].y1_int,aug_bbs[k].bounding_boxes[0].height,aug_bbs[k].bounding_boxes[0].width) for k in within_image_indices])
    aug_images = aug_images[within_image_indices,:,:,:]
                                              aug_boxes_list.append(aug_boxes)
                                                  aug_images_list.append(aug_images)
                                                    aug_images = np.vstack(tuple(aug_images_list))
                                                      aug_boxes = np.vstack(tuple(aug_boxes_list))
                                                        np.save(save_path + '/augmented_image.npy',aug_images)
                                                          np.save(save_path + '/augmented_boxes.npy',aug_boxes)
                                                            if load:
                                                                    return aug_images,aug_boxes
