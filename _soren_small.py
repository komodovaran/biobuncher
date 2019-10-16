import pandas as pd
import pims
import numpy as np

lip_int_size = 20
lip_BG_size = 45


def cmask(index, array, BG_size, int_size):
    a, b = index
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
    mask2 = x * x + y * y <= lip_int_size + 9  # to make a "gab" between BG and roi
    
    BG_mask = (x * x + y * y <= lip_BG_size)
    BG_mask = np.bitwise_xor(BG_mask, mask2)
    return (sum((array[mask]))), np.median(((array[BG_mask]))), mask, BG_mask

def signal_extractor(video, df, color):
    def extract(row):
        b, a = row['x'], row['y']
        frame = int(row['frame'])
        array = video[frame]
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x ** 2 + y ** 2 <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x ** 2 + y ** 2 <= lip_int_size + 9  # to make a "gab" between BG and roi
        BG_mask = (x ** 2 + y ** 2 <= lip_BG_size)
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

tracks = pd.read_csv(
    "results/_A_CLTA-TagRFP EGFP-Aux1-A7D2_Cell1_1s_TagRFP_GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_Cy3  TIRF Q-1.tif.csv")

vid1 = "data/A_CLTA-TagRFP EGFP-Aux1-A7D2/Cell1_1s/TagRFP/GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_Cy3  TIRF Q-1.tif"
vid1 = pims.TiffStack(vid1)[0:5]
full_tracked = signal_extractor(video = vid1, df = tracks, color = 'red')

