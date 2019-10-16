#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:16:28 2019

@author: sorensnielsen
"""
import os
import random
import time
from glob import glob

import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.ndimage
import seaborn as sns
import trackpy as tp
import pims
from pims import ImageSequence, ImageSequence, TiffStack
from pomegranate import *
from scipy import ndimage, ndimage
from scipy.stats.stats import pearsonr, pearsonr
from skimage import feature, feature, feature, io, measure, measure, measure
from skimage.color import rgb2gray, rgb2gray
from skimage.feature import blob_log, blob_log, blob_log, peak_local_max, peak_local_max

seaborn.set_style('whitegrid')
import itertools
import cython
import probfit
from matplotlib import ticker
from sklearn import neighbors
from tqdm import tqdm
import iminuit as Minuit

from probfit import BinnedLH, Chi2Regression, Extended, BinnedChi2, UnbinnedLH, gaussian
from matplotlib import patches


def image_loader_video(video):
    images_1 = TiffStack(video)
    return images_1


def step_tracker(df):
    microns_per_pixel = 0.160
    steps = []
    msd = []
    lag = []
    df['x'] = df['x'] * microns_per_pixel
    df['y'] = df['y'] * microns_per_pixel
    group_all = df.groupby('particle')
    x_step = []
    y_step = []
    
    # easiest: compute step in x, step in y and then steps
    for name, group in group_all:
        x_list = group.x.tolist()
        x_tmp = [y - x for x, y in zip(x_list, x_list[1:])]
        x_tmp.insert(0, 0.)
        
        y_list = group.y.tolist()
        y_tmp = [y - x for x, y in zip(y_list, y_list[1:])]
        y_tmp.insert(0, 0.)
        y_step.extend(y_tmp)
        x_step.extend(x_tmp)
        step_tmp = [np.sqrt(y ** 2 + x ** 2) for y, x in zip(y_tmp, x_tmp)]
        steps.extend(step_tmp)
    
    df['x_step'] = x_step
    df['y_step'] = y_step
    df['steplength'] = steps
    return df


def tracker(video, mean_multiplier, sep):
    full = tp.batch(video, 13, invert = False, minmass = mean_multiplier * np.mean(video[0]), separation = sep);
    full_tracked = tp.link_df(full, search_range, memory = 1)  # 5 pixel search range, memory =2
    full_tracked = tp.filter_stubs(full_tracked, 5)  # filter aour short stuff
    
    # check for subpixel accuracy
    tp.subpx_bias(full_tracked)
    plt.show()
    print(len(full_tracked))
    
    full_tracked = step_tracker(full_tracked)
    full_tracked['particle'] = full_tracked['particle'].transform(int)
    # full_tracked['particle'] = full_tracked['particle'].transform(str)
    full_tracked['duration'] = full_tracked.groupby('particle')['particle'].transform(len)
    
    def msd_df(df):
        max_lagtime = max(df['duration'])
        microns_per_pixel = 0.160
        frame_per_sec = float(1000 / 35.)
        df_msd = tp.imsd(df, microns_per_pixel, frame_per_sec, max_lagtime = max_lagtime)
        return df_msd
    
    msd_df = msd_df(full_tracked)
    msd_df = msd_df.stack().reset_index()
    msd_df.columns = ['time', 'particle', 'msd']
    return full_tracked, msd_df


def cmask(index, array, BG_size, int_size):
    a, b = index
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
    mask2 = x * x + y * y <= lip_int_size + 9  # to make a "gab" between BG and roi
    
    BG_mask = (x * x + y * y <= lip_BG_size)
    BG_mask = np.bitwise_xor(BG_mask, mask2)
    return (sum((array[mask]))), np.median(((array[BG_mask]))), mask, BG_mask


def signal_extractor(video, final_df, red_blue):  # change so taht red initial is after appearance timing
    
    # final_df = final_df.sort_values(['particle', 'frame'], ascending=True)
    
    def df_extractor2(row):
        b, a = row['x'], row['y']
        frame = int(row['frame'])
        array = video[frame]
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size + 9  # to make a "gab" between BG and roi
        BG_mask = (x * x + y * y <= lip_BG_size)
        BG_mask = np.bitwise_xor(BG_mask, mask2)
        
        return np.sum((array[mask])), np.min(((array[BG_mask])))  # added np in sum
    
    size_maker = np.ones(video[0].shape)
    ind = 25, 25  # dont ask - leave it here, it just makes sure the below runs
    mask_size, BG_size, mask, BG_mask = cmask(ind, size_maker, lip_BG_size, lip_int_size)
    mask_size = np.sum(mask_size)
    
    a = final_df.apply(df_extractor2, axis = 1)
    
    intensity = []
    bg = []
    for line in a:
        i, b = line
        bg.append(b)
        intensity.append(i)
    if red_blue == 'red' or red_blue == 'Red':
        final_df['red_int'] = intensity
        final_df['red_bg'] = bg
        final_df['red_int_corrected'] = final_df['red_int'] - (final_df['red_bg'] * mask_size)
    elif red_blue == 'Blue' or red_blue == 'blue':
        final_df['blue_int'] = intensity
        final_df['blue_bg'] = bg
        final_df['blue_int_corrected'] = final_df['blue_int'] - (final_df['blue_bg'] * mask_size)
    else:
        final_df['green_int'] = intensity
        final_df['green_bg'] = bg
        final_df['green_int_corrected'] = final_df['green_int'] - (final_df['green_bg'] * mask_size)
    
    return final_df


def plot_traces_on_img(df, save_path, cell_img):
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    points = np.array([[1, 2, 3, 4, 5, 2, 3, 4], [1, 2, 3, 4, 21, 31, 3, 4, 12, 31, 1, 2, 3]]).transpose().reshape(-1,
                                                                                                                   1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis = 1)
    t = np.linspace(0, 1, len([1, 2, 3, 4, 5, 2, 3, 4]))
    lc = LineCollection(segs, cmap = plt.get_cmap('inferno'))
    lc.set_array(t)
    
    fig, ax = plt.subplots(figsize = (10, 10))
    cm = plt.get_cmap('inferno')
    
    ax.imshow(cell_img[0], cmap = "ocean")
    from tqdm import tqdm
    group_all = df.groupby('particle')
    for name, group in tqdm(group_all):
        # x,y = return_normed_coordinates(group)
        x, y = return_coordinates(group)
        x = x * 1 / 0.160
        y = y * 1 / 0.160
        im = ax.set_prop_cycle('color', [cm(1. * i / (x.shape[-1] - 1)) for i in range(x.shape[-1] - 1)])
        for i in range(x.shape[-1] - 1):
            im = ax.plot([x[i + 1], x[i]], [y[i + 1], y[i]], alpha = 0.6)
    
    ax.set_ylabel('um', size = 12)
    ax.set_xlabel('um', size = 12)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 10)
    ax.grid(False)
    cbar = fig.colorbar(lc, ticks = [0, 0.5, 1])
    fig.tight_layout()
    fig.savefig(save_path + '__traces_on_img.pdf')


mean_multiplier = 2
sep = 3  # pixel separation
# Tracking parameters
memory = 1  # frame
search_range = 5  # pixels

lip_int_size = 20
lip_BG_size = 45

vid1 = '/Volumes/Soeren/Cell_studies/Joihannes/take2/Cell1_1s 2/TagRFP/GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_Cy3  TIRF Q-1.tif'

vid2 = '/Volumes/Soeren/Cell_studies/Joihannes/take2/Cell1_1s 2/EGFP/GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_GFP  TIRF Q-1.tif'

fret_vals = []
steplengths = []

vid1 = image_loader_video(vid1)
vid2 = image_loader_video(vid2)
full_tracked, msd_df = tracker(vid1, 5, 2)
full_tracked = full_tracked[full_tracked.duration > 20]

full_tracked = signal_extractor(vid1, full_tracked, 'red')
full_tracked = signal_extractor(vid2, full_tracked, 'green')
