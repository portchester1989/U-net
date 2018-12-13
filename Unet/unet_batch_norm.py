# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     U-net 
# File Name:        unet-Kares
# Date:             2/9/18 3:59 PM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/U-net
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
'''
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Dense, Softmax,BatchNormalization
from keras.models import *
from keras.optimizers import *
from keras.utils import to_categorical
from data_Keras import DataProcess
#from keras_yolo3.yolo3.model import *

class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_train_data(self,mode = 'my_data'):
        mydata = DataProcess(self.img_rows, self.img_cols)
        if mode == 'my_data':
            imgs_train, imgs_mask_train = mydata.load_my_train_data()
        elif mode == 'pretrain':
            imgs_train, imgs_mask_train = mydata.load_train_data()
        else:
            imgs_train, imgs_mask_train = mydata.load_small_train_data()
        imgs_mask_train = to_categorical(imgs_mask_train, num_classes=2)
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        mydata = DataProcess(self.img_rows, self.img_cols)
        imgs_test = mydata.load_test_data()
        return imgs_test
    
    def get_body(self,pretrained_weights=None):
        inputs = Input((256, 256, 3))
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        print(conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        print(conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print(pool1.shape)
        print('\n')
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        print(conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        print(conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print(pool2.shape)
        print('\n')
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        print(conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        print(conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print(pool3.shape)
        print('\n')
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        print(conv4.shape)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        print(conv4.shape)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        print(pool4.shape)
        print('\n')

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        print(conv5.shape)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        print(conv5.shape)
        drop5 = Dropout(0.5)(conv5)
        print('\n')
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        up6 = BatchNormalization()(up6)
        print(up6.shape)
        print(drop4.shape)
        merge6 = concatenate([drop4, up6], axis=3)
        print('merge: ')
        print(merge6.shape)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')( UpSampling2D(size=(2, 2))(conv6))
        up7 = BatchNormalization()(up7)
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')( UpSampling2D(size=(2, 2))(conv7))
        up8 = BatchNormalization()(up8)
        merge8 = concatenate([conv2, up8], axis=3) 
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')( UpSampling2D(size=(2, 2))(conv8))
        up9 = BatchNormalization()(up9)
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        #conv9 = Flatten(conv9)
        #conv10 = Dense(num_anchors ½I* 6)(conv9)
        #conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        conv10 = Softmax()(conv9)
        print(conv10.shape)
        model = Model(input=inputs,output=conv10)
        if pretrained_weights != None:
            model.load_weights(pretrained_weights)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        #print('model compile')
        return model
    def train(self,mode='pretrain',img_paths = None,bounding_box_path = None,save_path=None,imgs_train_pre=None,imgs_mask_train_pre=None,pretrained_weights=None):
        print("loading data")
        if mode == 'pretrain':
            imgs_train, imgs_mask_train = self.load_train_data()
        elif mode == 'from images': 
            imgs_train, imgs_mask_train = save_training_data(img_paths,bounding_box_path,save_path,load=True)
        elif mode == 'already': 
            imgs_train,imgs_mask_train = imgs_train_pre,imgs_mask_train_pre
        else: 
            imgs_train, imgs_mask_train = load_training_data(save_path)
        print("loading data done")
        model = self.get_body(pretrained_weights)
        print("got unet") # 保存的是模型和权重,
        model_checkpoint = ModelCheckpoint('../data_set/unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        early_call_back = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        tensor_board = TensorBoard(log_dir='./logs')
        print('Fitting model...')
        model.fit(x=imgs_train, y=imgs_mask_train, validation_split=0.2, batch_size=4, epochs=1, verbose=1, shuffle=True, callbacks=[model_checkpoint,early_call_back,tensor_board])

    def test(self):
        print("loading data")
        imgs_test = self.load_test_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model.load_weights('../data_set/unet.hdf5')
        print('predict test data')
        # imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        # np.save('../data_set/imgs_mask_test.npy', imgs_mask_test)

if __name__ == '__main__':
    unet = myUnet()
    unet.get_body()
    #unet.train(mode='already',img_train_pre=train_img,imgs_mask_train_pre=train_label)
    # unet.test()
