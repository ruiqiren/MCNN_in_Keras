# -*- coding:utf-8 _*-
"""
@author: steven.yi
@date: 2019/3/26
@file: test.py
@description: test
"""
from keras.models import load_model
from utils.data_loader import DataLoader
import numpy as np
import config as cfg
import sys
import os
import cv2


def save_density_map(save_dir, density_map, fname='results.png'):
    density_map = 255*density_map/np.max(density_map)
    density_map = density_map[0][0]
    cv2.imwrite(os.path.join(save_dir, fname), density_map)


if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    print('usage: python test.py A(or B)')
    exit()
print('Testing Part_{} ...'.format(dataset))

test_path = cfg.test_path.format(dataset)
test_gt_path = cfg.test_gt_path.format(dataset)

if dataset == 'A':
    model_path = './trained_models/mcnn_A_train.hdf5'
else:
    model_path = './trained_models/mcnn_B_train.hdf5'

output_dir = './output_{}/'.format(dataset)
density_maps_dir = os.path.join(output_dir, 'density_maps')
results_txt = os.path.join(output_dir, 'results.txt')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(density_maps_dir):
    os.mkdir(density_maps_dir)

model = load_model(model_path)
data_loader = DataLoader(test_path, test_gt_path, shuffle=False, gt_downsample=True)

mae = 0.0
mse = 0.0
for blob in data_loader:
    img = blob['data']
    gt = blob['gt']
    pred = model.predict(np.expand_dims(img, axis=0))
    gt_count = np.sum(gt)
    pred_count = np.sum(pred)
    mae += abs(gt_count - pred_count)
    mse += ((gt_count - pred_count) * (gt_count - pred_count))
    # save density map
    save_density_map(density_maps_dir, pred, blob['fname'].split('.')[0] + '.png')
    # save results
    with open(results_txt, 'a') as f:
        line = '<{}> {:.2f} -- {:.2f}\n'.format(blob['fname'].split('.')[0], gt_count, pred_count)
        f.write(line)

mae = mae/data_loader.num_samples
mse = np.sqrt(mse/data_loader.num_samples)
print('\nMAE: %0.2f, MSE: %0.2f' % (mae, mse))

with open(results_txt, 'a') as f:
    f.write('MAE: %0.2f, MSE: %0.2f' % (mae, mse))
