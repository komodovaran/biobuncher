#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:16:28 2019

@author: sorensnielsen
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage.io
import trackpy as tp
import pims
import numba

sns.set_style("whitegrid")


@numba.jit
def step_tracker(df):
    microns_per_pixel = 0.160
    steps = []
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
    full_tracked = tp.filter_stubs(full_tracked, 5)
    
    # check for subpixel accuracy
    # tp.subpx_bias(full_tracked)
    # plt.show()
    # print(len(full_tracked))
    #
    # full_tracked = step_tracker(full_tracked)
    # full_tracked['particle'] = full_tracked['particle'].transform(int)
    # # full_tracked['particle'] = full_tracked['particle'].transform(str)
    # full_tracked['duration'] = full_tracked.groupby('particle')['particle'].transform(len)
    #
    # def msd_df(df, microns_per_pixel, frames_per_sec, max_lagtime):
    #     df_msd = tp.imsd(traj = df, mpp = microns_per_pixel, fps = frames_per_sec, max_lagtime = max_lagtime)
    #     return df_msd
    #
    # msd_df = msd_df(full_tracked)
    # msd_df = msd_df.stack().reset_index()
    # msd_df.columns = ['time', 'particle', 'msd']
    return full_tracked#, msd_df


def cmask(index, array, BG_size, int_size):
    a, b = index
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    mask = x**2 + y**2 <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
    mask2 = x**2 + y**2 <= lip_int_size + 9  # to make a "gab" between BG and roi
    
    BG_mask = (x**2 + y**2 <= lip_BG_size)
    BG_mask = np.bitwise_xor(BG_mask, mask2)
    
    return (sum((array[mask]))), np.median(((array[BG_mask]))), mask, BG_mask


def signal_extractor(video, df, color):
    def extract(row):
        b, a = row['x'], row['y']
        frame = int(row['frame'])
        array = video[frame]
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x**2 + y**2 <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x**2 + y**2 <= lip_int_size + 9  # to make a "gab" between BG and roi
        BG_mask = (x**2 + y**2 <= lip_BG_size)
        BG_mask = np.bitwise_xor(BG_mask, mask2)
        return np.sum((array[mask])), np.min(((array[BG_mask])))  # added np in sum
    
    size_maker = np.ones(video[0].shape)
    ind = 25, 25  # dont ask - leave it here, it just makes sure the below runs
    mask_size, BG_size, mask, BG_mask = cmask(ind, size_maker, lip_BG_size, lip_int_size)
    mask_size = np.sum(mask_size)
    
    a = df.apply(extract, axis = 1)
    
    intensity = []
    bg = []
    for line in a:
        i, b = line
        bg.append(b)
        intensity.append(i)
    
    if color == 'red' or color == 'Red':
        df['{}_int'.format(color)] = intensity
        df['{}_bg'.format(color)] = bg
        df['{}_int_corrected'.format(color)] = intensity - (df['red_bg'] * mask_size)
    elif color == 'Blue' or color == 'blue':
        df['blue_int'] = intensity
        df['blue_bg'] = bg
        df['blue_int_corrected'] = df['blue_int'] - (df['blue_bg'] * mask_size)
    else:
        df['green_int'] = intensity
        df['green_bg'] = bg
        df['green_int_corrected'] = df['green_int'] - (df['green_bg'] * mask_size)
    
    return df


def plot_traces_on_img(df, save_path, cell_img):
    from matplotlib.collections import LineCollection
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

vid1 = "data/A_CLTA-TagRFP EGFP-Aux1-A7D2/Cell1_1s/TagRFP/GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_Cy3  TIRF Q-1.tif"
vid2 = "data/A_CLTA-TagRFP EGFP-Aux1-A7D2/Cell1_1s/EGFP/GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_GFP  TIRF Q-1.tif"

fret_vals = []
steplengths = []

vid1 = pims.TiffStack(vid1)[0:5]
vid2 = pims.TiffStack(vid2)[0:5]
full_tracked, msd_df = tracker(vid1, mean_multiplier = 5, sep = 2)
# full_tracked = full_tracked[full_tracked.duration > 20]

full_tracked = signal_extractor(video = vid1, df = full_tracked, color = 'red')
full_tracked = signal_extractor(video = vid2, df = full_tracked, color = 'green')
print(full_tracked)
# plot_traces_on_img(df = full_tracked, save_path = "results/", cell_img = skimage.io.imread(
#     "data/A_CLTA-TagRFP EGFP-Aux1-A7D2/Cell1_1s/TagRFP/GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_Cy3  TIRF Q-1.tif"))
