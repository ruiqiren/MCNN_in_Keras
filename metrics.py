# -*- coding:utf-8 _*-
"""
@author: steven.yi
@date: 2019/3/26
@file: metrics.py
@description: define the evaluation metrics
"""
import keras.backend as K


def mae(y_true, y_pred):
    return K.abs(K.sum(y_true) - K.sum(y_pred))


def mse(y_true, y_pred):
    return K.square(K.sum(y_true) - K.sum(y_pred))
