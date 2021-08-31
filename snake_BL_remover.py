# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:57:31 2021

@author: Mozaffarim

This is the code for 1D snake
"""

# I want to do peak finder idea again in 1D and with better data

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# there is an assumption that spectrum starts from zero with resolution of one.
# for this reason, we have an offset for just visualization of x_axis

def One_Dimensional_snake_BL_corrector(spectrum, parameters):
        
    from sklearn.preprocessing import minmax_scale
    from scipy.interpolate import interp1d

    end = parameters["BL_correct_mode"]
    iteration = parameters["iteration"]
    alpha = parameters["alpha"]
    gamma = parameters["gamma"]
    k = parameters["k"]
    conv_filter_size = parameters["smoothing_filter_size"]
    spectrum_range = len(spectrum)
    snake_distants = int(spectrum_range / parameters["number_of_snaxels"])
    
    min_range = 0
    max_range = spectrum_range - 1
    
    x = np.linspace(min_range, max_range, spectrum_range)

    x_visual_offset = parameters["x_visualization_offset"]    
    
    # convolve to smooth and then zero paddings are changed from zero to neighbours 
    edges = int((conv_filter_size - 1) / 2)
    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth 
    
    f_smoothed = smooth(spectrum, conv_filter_size)
    f_smoothed[:edges] = spectrum[:edges]
    f_smoothed[-edges:] = spectrum[-edges:]
    f = f_smoothed.copy()
    
    # normalizeing
    # f = minmax_scale(f, feature_range=(0, 1))
    x_s = x.copy()
    x_s = x_s[::snake_distants]
    snake_size = len(x_s)
    f_s = f.copy()
    f_s = f_s[::snake_distants]
    

    
    F_mg = np.zeros((snake_size))
    F_inclined = np.zeros((snake_size))
    F_spring = np.zeros((snake_size))
          
    G_1 = np.gradient(f[np.int16(x_s)], 1)
    G_2 = np.gradient(f[np.int16(x_s)], 2)        

    if parameters["visualization_mode"] == True:
        
        fig, axes = plt.subplots(3, 1, sharex='col', dpi=500, figsize=(10, 12), gridspec_kw = {'wspace':0, 'hspace':0.2}, linewidth=40)
        axes[0].plot(x + x_visual_offset, f, 'b', lw=1)
        axes[0].scatter(x_s + x_visual_offset, f_s, 40, color='green', marker='x', label="Initialized Snaxels")
        axes[0].bar(x_s + x_visual_offset, G_1, 10, color='blue', label="1st Gradient at Snaxels")
        axes[0].bar(x_s + x_visual_offset, G_2, 10, color='orange', label="2nd Gradient at Snaxels")
    
    area_under_the_curve = 1000000
    for j in range(iteration):
        for i in range(snake_size-1):
            
            if i == 0:
                if end.startswith('free'): 
                    x_s[i] = x_s[i + 1] + 1
                if end.startswith('fixed'):
                    continue
            if i == snake_size - 1:
                if end.endswith('free'):  
                    x_s[i] = x_s[i - 1] - 1
                if end.endswith('fixed'):
                    continue
 
            f_x_i = f[np.int16(x_s[i])]
            f_x_i_neg = f[np.int16(x_s[i - 1])]
            f_x_i_pos = f[np.int16(x_s[i + 1])]
            
            if f_x_i - f_x_i_neg > 0:
                F_mg_neg = (f_x_i - f_x_i_neg) / np.sqrt(((x_s[i] - x_s[i - 1]) ** 2) + ((f_x_i - f_x_i_neg) ** 2))
                F_mg[i] -= F_mg_neg
                
            elif f_x_i - f_x_i_pos > 0:
                F_mg_pos = (f_x_i - f_x_i_pos) / np.sqrt(((x_s[i] - x_s[i + 1]) ** 2) + ((f_x_i - f_x_i_pos) ** 2))
                F_mg[i] += F_mg_pos
                
            else:
                F_mg[i] = 0
                
            F_inclined[i] = G_1[i] + 0.5 * G_2[i] 
            
            x_old = x_s[i]
            
            x_s[i] = x_s[i] - alpha * F_mg[i] - gamma * F_inclined[i]

            F_spring[i] = -k * (x_s[i] - x_old)
            
            
            x_s[i] = x_s[i] + F_spring[i]
            
            if x_s[i] < min_range or x_s[i] > max_range:
                x_s[i] = (max_range - min_range) / 2
                                            
        G_1 = np.gradient(f[np.int16(x_s)], 1)
        G_2 = np.gradient(f[np.int16(x_s)], 2)         
        
        if parameters["Area_Convergence"]:
            xxx = x_s.astype(np.int16)
            xxx = np.unique(xxx)
            f_xxx = f[xxx]
            fnew = interp1d(xxx, f_xxx, kind=parameters["interpolation_mode"], fill_value='extrapolate', assume_sorted=True)
            area = np.sum(np.abs(fnew(x)))
        
            # I added this part to keep the result of the best iteration
            if area <= area_under_the_curve:
                area_under_the_curve = area
                x_s_opt = x_s.copy()
       
    if parameters["Area_Convergence"]:
        x_s = np.sort(x_s_opt)
    else:
        x_s = np.sort(x_s)
    
    if parameters["visualization_mode"] == True:
        axes[0].scatter(np.int16(x_s + x_visual_offset), f[np.int16(x_s)], 40, color='red', marker='x', label="Estimated Snaxels")
        axes[0].legend(loc='upper right')
    
    xxx = x_s.astype(np.int16)
    xxx = np.unique(xxx)
    f_xxx = f[xxx]
    
    fnew = interp1d(xxx, f_xxx, kind=parameters["interpolation_mode"], fill_value='extrapolate', assume_sorted=True)
    x = x.astype(np.int16)
    
    if parameters["visualization_mode"] == True:
        
        axes[1].plot(x + x_visual_offset, f, 'b', lw=1, label="Raw Input Spectrum")
        axes[1].plot(x + x_visual_offset, fnew(x), 'r', lw=2, label="Estimated Baseline")
        axes[1].legend(loc='upper right')
        
        axes[2].plot(x + x_visual_offset, f, 'b', lw=1, label="Raw Input Spectrum")
        axes[2].plot(x + x_visual_offset, f - fnew(x), 'k', lw=2, label="Baseline Corrected Spectrum")
        axes[2].legend(loc='upper right')
        plt.xlabel("Channel Index", size=12)
        
        axes[-1].set_ylabel('.', color=(0, 0, 0, 0))
        fig.text(0.05, 0.5, "Arbitrary Intensity", size=12, va='center', ha='center', rotation='vertical')
        
        plt.savefig("./Analysis_Image.png")
    raw_x = x.copy()
    raw_f = f.copy()
    x_snake = x_s.copy()
    baseline_corrected_f = f - fnew(x)
    baseline = fnew(x).copy()
    return raw_x, raw_f, x_snake, baseline_corrected_f, baseline

    