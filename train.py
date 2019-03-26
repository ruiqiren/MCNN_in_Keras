# -*- coding:utf-8 _*-
"""
@author: steven.yi
@date: 2019/3/22
@file: train.py
@description: 训练
"""
from keras.layers import Input
from keras.optimizers import Adam
from keras import metrics
from keras.callbacks import ModelCheckpoint
from model import MCNN
from utils.data_loader import DataLoader
import config as cfg
import sys
import os

if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    print('usage: python3 train.py A(or B)')
    exit()
print('Training Part_{} ...'.format(dataset))

train_path = cfg.train_path.format(dataset)
train_gt_path = cfg.train_gt_path.format(dataset)
val_path = cfg.val_path.format(dataset)
val_gt_path = cfg.val_gt_path.format(dataset)

input_shape = (None, None, 1)
model = MCNN(input_shape)
adam = Adam(lr=1e-4)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=[metrics.mae, metrics.mse])

train_data_gen = DataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True)
val_data_gen = DataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True)

checkpointer_best_val = ModelCheckpoint(
    filepath=os.path.join(cfg.output_dir, 'mcnn_'+dataset+'_val.hdf5'),
    monitor='val_loss', verbose=1, save_best_only=True, mode='min'
)
checkpointer_best_train = ModelCheckpoint(
    filepath=os.path.join(cfg.output_dir, 'mcnn_'+dataset+'_train.hdf5'),
    monitor='loss', verbose=1, save_best_only=True, mode='min'
)
callback_list = [checkpointer_best_train, checkpointer_best_val]

model.fit_generator(train_data_gen.flow(cfg.Train_Batch_Size),
                    steps_per_epoch=train_data_gen.num_samples // cfg.Train_Batch_Size,
                    validation_data=val_data_gen.flow(cfg.Val_Batch_Size),
                    validation_steps=val_data_gen.num_samples // cfg.Val_Batch_Size,
                    epochs=cfg.Epochs,
                    callbacks=callback_list,
                    verbose=1)
