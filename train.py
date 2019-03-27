# -*- coding:utf-8 _*-
"""
@author: steven.yi
@date: 2019/3/22
@file: train.py
@description: 训练
"""
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from model import MCNN
from utils.data_loader import DataLoader
import config as cfg
import sys
import os
from metrics import mae, mse


if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    print('usage: python3 train.py A(or B)')
    exit()

train_path = cfg.TRAIN_PATH.format(dataset)
train_gt_path = cfg.TEST_GT_PATH.format(dataset)
val_path = cfg.VAL_PATH.format(dataset)
val_gt_path = cfg.VAL_GT_PATH.format(dataset)

train_data_gen = DataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True)
val_data_gen = DataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True)

input_shape = (None, None, 1)
model = MCNN(input_shape)
adam = Adam(lr=1e-4)
model.compile(loss='mse', optimizer=adam, metrics=[mae, mse])

checkpointer_best_val = ModelCheckpoint(
    filepath=os.path.join(cfg.MODEL_DIR, 'mcnn_'+dataset+'_val.hdf5'),
    monitor='val_loss', verbose=1, save_best_only=True, mode='min'
)
checkpointer_best_train = ModelCheckpoint(
    filepath=os.path.join(cfg.MODEL_DIR, 'mcnn_'+dataset+'_train.hdf5'),
    monitor='loss', verbose=1, save_best_only=True, mode='min'
)
callback_list = [checkpointer_best_train, checkpointer_best_val]

# model.fit_generator(train_data_gen.flow(cfg.Train_Batch_Size),
#                     steps_per_epoch=train_data_gen.num_samples // cfg.Train_Batch_Size,
#                     validation_data=val_data_gen.flow(cfg.Val_Batch_Size),
#                     validation_steps=val_data_gen.num_samples // cfg.Val_Batch_Size,
#                     epochs=cfg.Epochs,
#                     callbacks=callback_list,
#                     verbose=1)

x_train, y_train = train_data_gen.get_all()
x_val, y_val = val_data_gen.get_all()

print('Training Part_{} ...'.format(dataset))
history = model.fit(
    x=x_train, y=y_train, batch_size=1, epochs=200,
    validation_data=(x_val, y_val),
    callbacks=callback_list
)
