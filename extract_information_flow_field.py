# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 10:46:05 2018

Script that can be run on a directory, calculates optical flow and extracts useful information from the flow field.

@author: Guido de Croon.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import time
import sys

def determine_optical_flow(prev_bgr, bgr, graphics= True):
    
    # *******************************************************************
    # TODO: In the !second! lecture on optical flow, study this function
    # and change the parameters below to investigate the trade-off between
    # accuracy and computational efficiency
    # *******************************************************************
    
    # convert the images to grayscale:
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY);
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY);
    
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # detect features:
    points_old = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params);
    
    # calculate optical flow
    points_new, status, error_match = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points_old, None, **lk_params)
    
    # filter the points by their status:
    points_old = points_old[status == 1];
    points_new = points_new[status == 1];
    
    flow_vectors = points_new - points_old;
    
    if(graphics):
        im = prev_bgr.copy();
        n_points = len(points_old);
        color = (0,255,0);
        for p in range(n_points):
            cv2.arrowedLine(im, tuple(points_old[p, :]), tuple(points_new[p,:]), color);
         
        plt.figure();
        plt.imshow(im);
        plt.title('Optical flow');
        #cv2.imshow('Flow', im);
        #cv2.waitKey(100);
        #cv2.destroyAllWindows()
    
    return points_old, points_new, flow_vectors;

def estimate_linear_flow_field(points_old, flow_vectors, RANSAC=False, n_iterations=100, error_threshold=10.0):
    
    n_points = points_old.shape[0];
    sample_size = 3; # minimal sample size is 3
    
    if(n_points >= sample_size):
        
        if(not RANSAC):
            
            # *****************************************
            # TODO: investigate this estimation method:
            # *****************************************
            
            # estimate a linear flow field for horizontal and vertical flow separately:
            # make a big matrix A with elements [x,y,1]
            A = np.concatenate((points_old, np.ones([points_old.shape[0], 1])), axis=1);
            
            # Moore-Penrose pseudo-inverse:
            # https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
            pseudo_inverse_A = np.linalg.pinv(A);
            
            # target = horizontal flow:
            u_vector = flow_vectors[:,0];
            # solve the linear system:
            pu = np.dot(pseudo_inverse_A, u_vector);
            # calculate how good the fit is:
            errs_u = np.abs(np.dot(A, pu) - u_vector);
            
            # target = vertical flow:
            v_vector = flow_vectors[:,1];
            pv = np.dot(pseudo_inverse_A, v_vector);
            errs_v = np.abs(np.dot(A, pv) - v_vector);
            err = (np.mean(errs_u) + np.mean(errs_v)) / 2.0;
            
        else:
            # This is a RANSAC method to better deal with outliers
            # matrices and vectors for the big system:
            A = np.concatenate((points_old, np.ones([points_old.shape[0], 1])), axis=1);
            u_vector = flow_vectors[:,0];
            v_vector = flow_vectors[:,1];
            
            # solve many small systems, calculating the errors:
            errors = np.zeros([n_iterations, 2]);
            pu = np.zeros([n_iterations, 3])
            pv = np.zeros([n_iterations, 3])
            for it in range(n_iterations):
                inds = np.random.choice(range(n_points), size=sample_size, replace=False);
                AA = np.concatenate((points_old[inds,:], np.ones([sample_size, 1])), axis=1);
                pseudo_inverse_AA = np.linalg.pinv(AA);
                # horizontal flow:
                u_vector_small = flow_vectors[inds, 0];
                # pu[it, :] = np.linalg.solve(AA, UU);
                pu[it,:] = np.dot(pseudo_inverse_AA, u_vector_small);
                errs = np.abs(np.dot(A, pu[it,:]) - u_vector);
                errs[errs > error_threshold] = error_threshold;
                errors[it, 0] = np.mean(errs);
                # vertical flow:
                v_vector_small = flow_vectors[inds, 0];
                # pv[it, :] = np.linalg.solve(AA, VV);
                pv[it, :] = np.dot(pseudo_inverse_AA, v_vector_small);
                errs = np.abs(np.dot(A, pv[it,:]) - v_vector);
                errs[errs > error_threshold] = error_threshold;
                errors[it, 1] = np.mean(errs);
            
            # take the minimal error
            errors = np.mean(errors, axis=1);
            ind = np.argmin(errors);
            err = errors[ind];
            pu = pu[ind, :];
            pv = pv[ind, :];
    else:
        # not enough samples to make a linear fit:
        pu = np.asarray([0.0]*3);
        pv = np.asarray([0.0]*3);
        err = error_threshold;
        
    return pu, pv, err;

# these functions are to get a nice directory listing
def get_number_file_name(name):
    inds1 = [m.start() for m in re.finditer('_', name)]
    if(inds1 == []):
        return 0;
    ind1 = inds1[-1];
    inds2 = [m.start() for m in re.finditer('\.', name)]
    if(inds2 == []):
        return 0;
    ind2 = inds2[-1];
    number = name[ind1+1:ind2];
    return int(number);

def compare_file_names(name1, name2):
    number1 = get_number_file_name(name1);
    number2 = get_number_file_name(name2);
    return number1 - number2;
    
def show_flow(image_nr_1, image_nr_2, image_dir_name = './image_sequence_pure_ver1/', image_prefix='image_', image_type = 'jpg'):
    image_name_1 = image_dir_name + image_prefix + str(image_nr_1) + '.' + image_type;
    prev_bgr = cv2.imread(image_name_1);
    image_name_2 = image_dir_name + image_prefix + str(image_nr_2) + '.' + image_type;
    bgr = cv2.imread(image_name_2);
    # print('name1: {}\nname2: {}'.format(image_name_1, image_name_2));
    points_old, points_new, flow_vectors = determine_optical_flow(prev_bgr, bgr, graphics=True);
    return points_old, points_new, flow_vectors;
    

def extract_flow_information(image_dir_name = './image_sequence_pure_ver1/', image_type = 'jpg', verbose=True, graphics = True, flow_graphics = False):
    
    # get the image names from the directory:
    image_names = [];
    for file in os.listdir(image_dir_name):
        if file.endswith(image_type):
            image_names.append(image_dir_name + file);
    if sys.version_info[0] < 3:
        # Python 2:
        image_names.sort(cmp=compare_file_names);
    else:
        # Python 3:
        image_names.sort(key=get_number_file_name);
    
    # iterate over the images:
    n_images = len(image_names);
    FoE_over_time = np.zeros([n_images, 2]);
    horizontal_motion_over_time = np.zeros([n_images, 1]);
    vertical_motion_over_time = np.zeros([n_images, 1]);
    divergence_over_time = np.zeros([n_images, 1]);
    errors_over_time = np.zeros([n_images, 1]);
    elapsed_times = np.zeros([n_images,1]);
    FoE = np.asarray([0.0]*2);
    for im in np.arange(0, n_images, 1):
        
        bgr = cv2.imread(image_names[im]);
        
        if(im > 0):
            
            t_before = time.time()
            # determine optical flow:
            points_old, points_new, flow_vectors = determine_optical_flow(prev_bgr, bgr, graphics=flow_graphics);
            # do stuff
            elapsed = time.time() - t_before;
            print('Elapsed time = {}'.format(elapsed));
            elapsed_times[im] = elapsed;

            # convert the pixels to a frame where the coordinate in the center is (0,0)
            points_old -= 128.0;
            
            # extract the parameters of the flow field:
            pu, pv, err = estimate_linear_flow_field(points_old, flow_vectors);
            
            # ************************************************************************************
            # TODO: assignment: extract the focus of expansion and divergence from the flow field:
            # ************************************************************************************
            # change the following five lines of code!
            horizontal_motion = -pu[2]; # 0.0;
            vertical_motion = -pv[2]; #0.0;
            # theoretically correct, but numerically not so stable:
            FoE[0] = -pu[2]/pu[0]; #0.0;
            FoE[1] = -pv[2]/pv[1]; #0.0;
            divergence = (pu[0]+pv[1]) / 2.0; # 0.0;
            
            # book keeping:
            horizontal_motion_over_time[im] = horizontal_motion;
            vertical_motion_over_time[im] = vertical_motion;
            FoE_over_time[im, 0] = FoE[0];
            FoE_over_time[im, 1] = FoE[1];
            divergence_over_time[im] = divergence;
            errors_over_time[im] = err;
            
            if(verbose):
                # print the FoE and divergence:
                print('error = {}, FoE = {}, {}, and divergence = {}'.format(err, FoE[0], FoE[1], divergence));
        
        # the current image becomes the previous image:
        prev_bgr = bgr;
    
    print('*** average elapsed time = {} ***'.format(np.mean(elapsed_times[1:,0])));
    
    if(graphics):
        
        # ********************************************************************
        # TODO:
        # What is the unit of the divergence?
        # Can you also draw the time-to-contact over time? In what unit is it?
        # ********************************************************************
        
        plt.figure();
        plt.plot(range(n_images), divergence_over_time, label='Divergence');
        plt.xlabel('Image')
        plt.ylabel('Divergence')
        
        plt.figure();
        plt.plot(range(n_images), FoE_over_time[:,0], label='FoE[0]');
        plt.plot(range(n_images), FoE_over_time[:,1], label='FoE[1]');
        plt.plot(range(n_images), np.asarray([0.0]*n_images), label='Center of the image')
        plt.legend();
        plt.xlabel('Image')
        plt.ylabel('FoE')
        
        plt.figure();
        plt.plot(range(n_images), errors_over_time, label='Error');
        plt.xlabel('Image')
        plt.ylabel('Error')
        
        plt.figure();
        plt.plot(range(n_images), horizontal_motion_over_time, label='Horizontal motion');
        plt.plot(range(n_images), vertical_motion_over_time, label='Vertical motion');
        plt.xlabel('Image')
        plt.ylabel('Motion U/Z')        

if __name__ == '__main__':        
    # extract_flow_information(image_dir_name = './images_landing/', image_type = 'png', verbose=True, graphics = True, flow_graphics = True);
    # Change flow_gaphics to True in order to see images and optical flow:
    extract_flow_information(flow_graphics=False)