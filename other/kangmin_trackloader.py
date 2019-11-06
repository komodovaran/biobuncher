import scipy.io
import pandas as pd
from itertools import chain
from itertools import groupby
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import streamlit as st

# pullMatrixData - function
# 
# INPUTS: ----------------------------------------------------------------------
#
# cellNum - integer | used to specify which folder from cmeAnalysis will be used
#           ex. Cell4_1s <-- cellNum = 4
#
# parentFolderPath - string | absolute / relative path to the parent folder containing the 'Cell1_1s' folders
#
# numConsecPVal - int | the number of consecutive p-values to classify as "Aux+"
#                  DEFAULT - 3
#
# pvalCutOff - float | p-value used as the cut-off for significance
#              DEFAULT - .005
#
# mustBeSecondHalf - boolean | constrains consecutive pvalues to fall into the second half of the lifetime
#                    DEFAULT - True
#
# OUTPUTS: ---------------------------------------------------------------------
#
# pd.dataframe | returns a dataframe containing each track as a row and the following as columns:
#        'frame', 'lifetime', 'max_intensity', 'background', 'totaldisp', 'max_msd', 'catIdx', 'aux', 
#        'avg_rise', 'avg_dec', 'risevsdec', 'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom'

def pullMatrixData(cellNum, parentFolderPath, numConsecPVal = 3, pvalCutOff = .005, mustBeSecondHalf = True, checkLifetime = True):
    #Load .mat tracks file
    mat = scipy.io.loadmat(parentFolderPath + 'Cell' + str(cellNum) + '_1s/TagRFP/Tracking/ProcessedTracks.mat')
    tracks = mat['tracks'][0,:]

    #Initialize DF
    df = pd.DataFrame(columns=['trackNum', 'frame', 'lifetime', 'max_intensity', 'background', 'totaldisp', 'max_msd', 'catIdx', 'aux', 'avg_rise', 'avg_dec', 'risevsdec', 'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom'])

    for i in range(len(tracks)):
        #TrackNum:
        trackNum = str(cellNum) + "-" + str(i + 1)

        #Intensity (A)
        intensity = tracks[i][4]
        intensity_t1 = list(intensity[0])
        intensity_t2 = list(intensity[1])
        intensity_max = max(intensity_t1)
        lifetime = len(intensity_t1)
        max_index = intensity_t1.index(intensity_max)

        #Background(c)
        bkground = tracks[i][5]
        bkground_t1 = list(bkground[0])
        bkground_t2 = list(bkground[1])
        bkground_scal = bkground_t1[max_index]

        #Category(catldx)
        catldx = list(chain.from_iterable(tracks[i][33]))
        catldx_scal = catldx[0]

        #Pvals(pvals)
        pvals = tracks[i][12]
        pvals_t2 = list(pvals[1])
        
        #Frame(f)
        frame = list(chain.from_iterable(tracks[i][1]))
        frame_scal = frame[max_index]

        #Start Buffer(startBuffer)
        startBuffer = list(chain.from_iterable(tracks[i][24]))

        #End Buffer(endBuffer)
        endBuffer = list(chain.from_iterable(tracks[i][25]))

        #Motion Analysis:
        motion_analysis = list(chain.from_iterable(tracks[i][26]))
        if not motion_analysis: #If motion analysis is empty...
            tdisp_scal = None
            msd_max = None
        else:
            tdisp = list(chain.from_iterable(motion_analysis[0][0]))
            tdisp_scal = tdisp[0]
            msd = list(chain.from_iterable(motion_analysis[0][1]))
            msd_max = max(msd)

        #Check if Aux+ or Aux-
        # Add classification parameters to this section
        isAux = False
        pval_cutoff = pvalCutOff
        #Look for Consecutive True's
        if(lifetime < 20 and checkLifetime):
            numConsecPValTemp = numConsecPVal - 1
        else:
            numConsecPValTemp = numConsecPVal
        for x in range(len(pvals_t2) - (numConsecPValTemp - 1)):
            isSig = False
            for j in range(x, x + (numConsecPValTemp)):
                isSig = pvals_t2[j] <= pval_cutoff
                if not isSig:
                    break
            if mustBeSecondHalf:
                if (x < (int(lifetime)/2)):
                    isSig = False
            if isSig:
                isAux = True
        if isAux:
            # Classify as Aux +
            aux = 1
        else:
            # Classify as Aux -
            aux = 0

        #Generate Full Intensity for Min / Max Rise / Decay
        if startBuffer and endBuffer:
            intensity_list = list(startBuffer[0][3][0]) + intensity_t1 + list(endBuffer[0][3][0])
            start_index = 1
            end_index = len(intensity_list)-1
        else:
            intensity_list = [0] + intensity_t1 + [0]
            start_index = intensity_list.index(intensity_t1[0])
            end_index = intensity_list.index(intensity_t1[-1])
        rise_slope = list()
        decay_slope = list()
        total_slope = list()
        for j in range(start_index, end_index + 1):
            slope = intensity_list[j] - intensity_list[j - 1]
            if slope >= 0:
                rise_slope.append(slope)
            else:
                decay_slope.append(slope)
            total_slope.append(slope)
        if len(rise_slope) == 0 and len(decay_slope) == 0:
            avgRise = 0
            avgDecay = 0
            riseVsDecay = 0
            avgRiseMom = 0
            avgDecMom = 0
            riseVsDecayMom = 0
        elif len(rise_slope) == 0:   
            avgRise = 0
            avgDecay = sum(decay_slope) / len(decay_slope)
            riseVsDecay = 0
            avgRiseMom = 0
            avgDecMom = 0
            riseVsDecayMom = 0
        elif len(decay_slope) == 0:   
            avgRise = sum(rise_slope) / len(rise_slope)
            avgDecay = 0
            riseVsDecay = 0
            avgRiseMom = 0
            avgDecMom = 0
            riseVsDecayMom = 0
        else:   
            avgRise = sum(rise_slope) / len(rise_slope)
            avgDecay = sum(decay_slope) / len(decay_slope)
            riseVsDecay = len(rise_slope) / len(decay_slope)

            #Get Moments
            rise_slope_mom = list()
            decay_slope_mom = list()
            start_index = 1
            end_index = len(total_slope)-1
            for j in range(start_index, end_index + 1):
                moment = total_slope[j] - total_slope[j - 1]
                if moment >= 0:
                    rise_slope_mom.append(moment)
                else:
                    decay_slope_mom.append(moment)
            if len(rise_slope_mom) == 0 and len(decay_slope_mom) == 0:
                avgRiseMom = 0
                avgDecMom = 0
                riseVsDecayMom = 0
            elif len(rise_slope_mom) == 0:   
                avgRiseMom = 0
                avgDecMom = 0
                riseVsDecayMom = 0
            elif len(decay_slope_mom) == 0:   
                avgRiseMom = 0
                avgDecMom = 0
                riseVsDecayMom = 0
            else:   
                avgRiseMom = sum(rise_slope_mom) / len(rise_slope_mom)
                avgDecayMom = sum(decay_slope_mom) / len(decay_slope_mom)
                riseVsDecayMom = len(rise_slope_mom) / len(decay_slope_mom)
            #endif
        #endif

        #Create Row
        row = [trackNum, frame_scal, lifetime, intensity_max, bkground_scal, tdisp_scal, msd_max, catldx_scal, aux, avgRise, avgDecay, riseVsDecay, avgRiseMom, avgDecayMom, riseVsDecayMom]
        
        # Add to DF
        df.loc[i] = row

    #End For Loop
    return (df)

# filterMatrixData - function
# 
# INPUTS:
#
# tom_data - pd.dataframe | a dataframe with rows as tracks and columns with all / some of the following:
#        'frame', 'lifetime', 'max_intensity', 'background', 'totaldisp', 'max_msd', 'catIdx', 'aux', 
#        'avg_rise', 'avg_dec', 'risevsdec', 'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom'
#
# dropNA - boolean | whether or not to drop tracks with NA (NaN / null) values.
#          DEFAULT - False
#
# catsToUse - array | values 1-8 that determine which categories to use
#             DEFAULT - all cats: [1,2,3,4,5,6,7,8]
#
# colsToDrop - array of strings | array of column header names (i.e. 'frame', 'lifetime', etc.) to be dropped
#              DEFAULT - empty array [] (no columns)
#

def filterMatrixData(data, dropNA = False, catsToUse = [1,2,3,4,5,6,7,8], colsToDrop = []):
    if dropNA:
        data = data.dropna()
    data = data.loc[data['catIdx'].isin(catsToUse)]
    for col in colsToDrop:
        data = data.drop(labels = col, axis = 1)
    return(data)

if __name__ == "__main__":
    @st.cache
    def _get_data():
        df = pullMatrixData(cellNum = 1, parentFolderPath = "/Users/Johannes/Desktop/PythonScripts/kangmin_data/")
        return df
    df = _get_data()
    st.write(df.head())

    for i, (_, group )in enumerate(df.groupby("trackNum")):
        print(group)
        if i == 5:
            break