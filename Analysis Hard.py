# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:18:52 2021

@author: woutg
"""
# Importing needed packages:
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, os.path
from sys import exit
import scipy.signal as sc

def point_rotator(vector, angle, middle):
    # This function rotates vectors around a middle point given a certain angle
    rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    vector_rotated = middle + rot_matrix.dot(vector-middle)
    return vector_rotated

def image_rotator(im, clipmargin=0.5, before_nozzle=5000, rod_condition=5e3, tube_condition=-3e3, streamcondition=250,
            dx_ref1=150, dx_ref2=-400, refhalfbandwidth=5, plot=False):
    # This function takes a stream image, selects two points in the middle
    # of that stream and then rotates it such that the stream falls straight.
    im_bw = np.sum(np.array(im), axis=2) # Image is turned into greyscale numpy array
    im_clipped = np.clip(im_bw, 0., clipmargin * np.max(im_bw)) # Clipped version -> no noise in clipped dark parts
    im_shape = np.shape(im_bw)

    # (Empty) arrays are prepared:
    x_array = np.arange(0, im_shape[1])
    y_array = np.arange(0, im_shape[0])

    hor_profile = np.zeros(im_shape[1])
    refp1_profile = np.zeros(im_shape[0])
    refp2_profile = np.zeros(im_shape[0])


    # The image is summed vertically to find rough horizontal profile:
    for i in range(im_shape[1]):
        hor_profile[i] = np.sum(im_clipped[:, i])
    hor_profile_diff = np.diff(hor_profile)

    # This horizontal profile is used to find the rod and nozzle locations:
    rod_mask = np.where(hor_profile_diff[:before_nozzle] > rod_condition)
    rod_top = int((np.min(rod_mask) + np.max(rod_mask)) / 2)

    tube_mask = np.where(hor_profile_diff[rod_top:] < tube_condition)
    tube_handle = rod_top + int((np.min(tube_mask) + np.max(tube_mask)) / 2)

    # x coords are defined wrt the rod and nozzle as points which can be used to straighten the image:
    refpoint_1_x = rod_top + dx_ref1
    refpoint_2_x = tube_handle + dx_ref2

    # Vertical profile is obtained in region around x coords of repoints to find the edges of the stream:
    for i in range(im_shape[0]):
        refp1_profile[i] = np.sum(im_clipped[i, refpoint_1_x - refhalfbandwidth:refpoint_1_x + refhalfbandwidth])
        refp2_profile[i] = np.sum(im_clipped[i, refpoint_2_x - refhalfbandwidth:refpoint_2_x + refhalfbandwidth])
    refp1_profile_diff = np.diff(refp1_profile)
    refp2_profile_diff = np.diff(refp2_profile)

    # The y-coords (middle of the stream) are calculated for the given x-coords of the refpoints:
    stream_refp1_left_mask = np.where(refp1_profile_diff < -streamcondition)
    stream_refp1_left = np.min(stream_refp1_left_mask)
    stream_refp1_right_mask = np.where(refp1_profile_diff > streamcondition)
    stream_refp1_right = np.max(stream_refp1_right_mask)
    stream_refp1 = (stream_refp1_left + stream_refp1_right) / 2

    stream_refp2_left_mask = np.where(refp2_profile_diff < -streamcondition)
    stream_refp2_left = np.min(stream_refp2_left_mask)
    stream_refp2_right_mask = np.where(refp2_profile_diff > streamcondition)
    stream_refp2_right = np.max(stream_refp2_right_mask)
    stream_refp2 = (stream_refp2_left + stream_refp2_right) / 2

    refpoint_1_y = (stream_refp1_left + stream_refp1_right)/2
    refpoint_2_y = (stream_refp2_left + stream_refp2_right)/2

    # We now have two refpoints in the vertical center of the stream at the horizontal top and bottom:
    refpoint_1, refpoint_2 = (refpoint_1_x, refpoint_1_y), (refpoint_2_x, refpoint_2_y)

    # We can use these refpoints to straighten the image:
    angle_stream = np.degrees(np.arctan((stream_refp2 - stream_refp1) / (refpoint_2_x - refpoint_1_x)))
    im_rotated = im.rotate(angle_stream)

    if plot:
        # If we want to check everything went well, we plot
        plt.plot(x_array, hor_profile)
        plt.plot(x_array[1:], 100 * hor_profile_diff)
        plt.vlines([rod_top, tube_handle], 0., 1.7e6, color='blue')
        plt.vlines([refpoint_1_x, refpoint_2_x], 0., 1.7e6, color='red')
        plt.hlines(-3e5, 0., 6000)
        # plt.xlim(5100, tube_handle)
        plt.xlabel('x coord [pix]')
        plt.ylabel('profile (value, 100*value for diff)')
        plt.title('Selected refpoints on horizontal profile')
        plt.show()

        plt.figure(figsize=(30, 20))
        plt.imshow(im_bw, cmap='gray')
        plt.vlines([rod_top, tube_handle], 0, 4000, color='blue', linewidth=5)
        plt.vlines([refpoint_1_x, refpoint_2_x], 0, 4000, color='red', linewidth=5)
        plt.plot([refpoint_1_x, refpoint_2_x], [refpoint_1_y, refpoint_2_y], color='green', linewidth=5)
        plt.xlabel('x coord [pix]')
        plt.ylabel('y coord [pix]')
        plt.title('Selected refpoints on image')
        plt.show()

        plt.plot(y_array, refp1_profile, color='red', label='Profile 1')
        plt.plot(y_array[1:], 10 * refp1_profile_diff, color='red', linestyle='dotted')
        plt.vlines([stream_refp1_left, stream_refp1, stream_refp1_right], -6000, 6000, color='red', linestyle='dashed')
        plt.plot(y_array, refp2_profile, color='blue', label='Profile 2')
        plt.plot(y_array[1:], 10 * refp2_profile_diff, color='blue', linestyle='dotted')
        plt.vlines([stream_refp2_left, stream_refp2, stream_refp2_right], -6000, 6000, color='blue', linestyle='dashed')
        plt.xlim(stream_refp2_left-100, stream_refp2_right+100)
        plt.xlabel('y coord [pix]')
        plt.ylabel('profile (value, 10*value for diff)')
        plt.title('Selected cross sections on image')
        plt.legend()
        plt.show()

        plt.figure(figsize=(30, 20))
        plt.imshow(im_rotated)
        plt.xlabel('x coord [pix]')
        plt.ylabel('y coord [pix]')
        plt.title('Rotated image')
        plt.show()

        print('The stream has an angle of ' + str(angle_stream) + ' degrees.')

    return (im_rotated, angle_stream, refpoint_1, refpoint_2, rod_top)

def edge_finder(image_rotated_bw, image_refpoint_1, image_refpoint_2, image_angle, middle, stream_searchwidth=300,\
                edgeband_width=7, moving_average_size=10, plot=False, extensiveplot=False):

    # This function takes an image of a stream, assumes it is straight,
    # i.e. that the vector of gravity points towards the left, and calculates the stream profile

    # Again, clipping is performed to reduce noise. Trimmed means clipped to a lesser degree:
    image_clipped = np.clip(image_rotated_bw, 0.4 * np.max(image_rotated_bw), 0.5 * np.max(image_rotated_bw))
    image_trimmed = np.clip(image_rotated_bw, 0.2 * np.max(image_rotated_bw), 0.6 * np.max(image_rotated_bw))

    # The reference points are rotated to match the rotated, straight, image:
    image_refpoint_1_rotated = point_rotator(image_refpoint_1, np.deg2rad(image_angle), middle).astype(int)
    image_refpoint_2_rotated = point_rotator(image_refpoint_2, np.deg2rad(image_angle), middle).astype(int)

    if extensiveplot:
        # If wanted, steps can be plotted:
        plt.figure(figsize=(30, 20))
        plt.imshow(image_clipped)
        plt.vlines([image_refpoint_1_rotated[0], image_refpoint_2_rotated[0]], 0, 4000, color='red',
                   linewidth=5)
        plt.plot([image_refpoint_1_rotated[0], image_refpoint_2_rotated[0]],
                 [image_refpoint_1_rotated[1], image_refpoint_2_rotated[1]], color='red', linewidth=5)
        plt.show()

    # Arrays are prepared:
    x_array_profile = np.arange(image_refpoint_1_rotated[0] - 150, image_refpoint_2_rotated[0] + 205)
    # We look 150 x-coords underneath refpoint 1 and 205 x-coords above repoint 2
    profile_left = np.zeros(355 + image_refpoint_2_rotated[0] - image_refpoint_1_rotated[0])
    profile_right = np.zeros(355 + image_refpoint_2_rotated[0] - image_refpoint_1_rotated[0])
    profile_left_course = np.zeros(355 + image_refpoint_2_rotated[0] - image_refpoint_1_rotated[0])
    profile_right_course = np.zeros(355 + image_refpoint_2_rotated[0] - image_refpoint_1_rotated[0])

    # For every height, the profile is calculated across the stream:
    for i in range(len(profile_left)):
        # Derivatives are calculated to find the edge.
        diff_clipped = np.diff(image_clipped[:, image_refpoint_1_rotated[0] - 150 + i])
        diff_trimmed = np.diff(image_trimmed[:, image_refpoint_1_rotated[0] - 150 + i])

        if i == 200 and extensiveplot:
            # If needed, the profile and its derivative is plotted:
            plt.plot(image_clipped[:, image_refpoint_1_rotated[0] - 150 + i][
                     image_refpoint_1_rotated[1] - stream_searchwidth:image_refpoint_1_rotated[1] + stream_searchwidth])
            plt.plot(image_trimmed[:, image_refpoint_1_rotated[0] - 150 + i][
                     image_refpoint_1_rotated[1] - stream_searchwidth:image_refpoint_1_rotated[1] + stream_searchwidth])
            plt.plot(10 * diff_clipped[image_refpoint_1_rotated[1] - stream_searchwidth:image_refpoint_1_rotated[
                                                                                            1] + stream_searchwidth])
            plt.plot(10 * diff_trimmed[image_refpoint_1_rotated[1] - stream_searchwidth:image_refpoint_1_rotated[
                                                                                            1] + stream_searchwidth])
            plt.vlines([stream_searchwidth], min(diff_clipped), max(diff_clipped), color='red')
            plt.show()

        # A course search of the edge is made by just searching for the maximal point in the gradient:
        stream_left_coarse = image_refpoint_1_rotated[1] - stream_searchwidth + np.argmin(\
            diff_clipped[image_refpoint_1_rotated[1] - stream_searchwidth:image_refpoint_1_rotated[1]])
        stream_right_coarse = image_refpoint_1_rotated[1] + np.argmax(diff_clipped[image_refpoint_1_rotated[1]: \
            image_refpoint_1_rotated[1] + stream_searchwidth])

        # If the point is too dissimilar to its neighbouring edge points,
        # it is considered wrong and we look for the next maximal gradient.
        if i != 0:
            difference_left = np.abs(stream_left_coarse - profile_left[i - 1])
            difference_right = np.abs(stream_right_coarse - profile_right[i - 1])

            if difference_left > 20:
                # If the point differs by more than 20 coords, it is considered wrong.
                diff_clipped[stream_left_coarse] = 0
                stream_left_coarse = image_refpoint_1_rotated[1] - stream_searchwidth + np.argmin( \
                    diff_clipped[image_refpoint_1_rotated[1] - stream_searchwidth:image_refpoint_1_rotated[1]])

            if difference_right > 20:
                diff_clipped[stream_right_coarse] = 0
                stream_right_coarse = image_refpoint_1_rotated[1] + np.argmax(diff_clipped[image_refpoint_1_rotated[1]:\
                                                                    image_refpoint_1_rotated[1] + stream_searchwidth])
        # Next, to calculate the location of the edge of the stream more precisely,
        # a narrow band is defined around the edge in which we will perform our analysis:
        stream_left_edgeband = diff_trimmed[stream_left_coarse - edgeband_width:stream_left_coarse + edgeband_width]
        stream_right_edgeband = diff_trimmed[stream_right_coarse - edgeband_width:stream_right_coarse + edgeband_width]

        sum_left = np.sum(stream_left_edgeband)
        sum_right = np.sum(stream_right_edgeband)

        # The refined edge location is found by calculating the "center of mass"
        # with the "mass" here being the gradient amplitude. This assumes we have a gaussian blurr on the image:
        if sum_left == 0:
            # This deals with rare errors where the course edge was still not right
            stream_left = stream_left_coarse
        else:
            # Calculation of the "center of mass"
            stream_left = stream_left_coarse + np.sum(
                np.arange(-edgeband_width, edgeband_width) * stream_left_edgeband) / sum_left

        if sum_right == 0:
            # This deals with rare errors where the course edge was still not right
            stream_right = stream_right_coarse
        else:
            # Calculation of the "center of mass"
            stream_right = stream_right_coarse + np.sum(
                np.arange(-edgeband_width, edgeband_width) * stream_right_edgeband) / sum_right

        # Arrays are filled with found values
        profile_left[i], profile_right[i] = stream_left, stream_right
        profile_left_course[i], profile_right_course[i] = stream_left_coarse, stream_right_coarse

    # To further smooth the profiles, a moving average is taken.
    # This is done in order to later find the wavelength for the smallest of amplitudes.
    profile_left_smooth = np.convolve(profile_left, np.ones(moving_average_size) / moving_average_size, mode='same')
    profile_right_smooth = np.convolve(profile_right, np.ones(moving_average_size) / moving_average_size, mode='same')

    # The middle is defined
    profile_middle = (profile_left + profile_right)/2

    if extensiveplot:
        plt.figure(figsize=(15, 10))
        plt.plot(profile_left_smooth, color='red', linewidth=3)
        plt.plot(profile_right_smooth, color='red', linewidth=3)
        plt.plot(profile_left_course, color='blue')
        plt.plot(profile_right_course, color='blue')
        #    plt.xlim(0, 200)
        #    plt.ylim(1780, 1820)
        plt.show()

    if plot:
        plt.figure(figsize=(15, 10), dpi=400)
        plt.imshow(image_rotated_bw, cmap='gray')
        plt.plot(x_array_profile, profile_left_smooth, color='red', linewidth=0.5)
        plt.plot(x_array_profile, profile_middle, color='orange', linewidth=0.5)
        plt.plot(x_array_profile, profile_right_smooth, color='blue', linewidth=0.5)
        plt.vlines([image_refpoint_1_rotated[0] - 150 + 123], 0, 4000)
        plt.scatter([image_refpoint_1_rotated[0], image_refpoint_2_rotated[0]],
                [image_refpoint_1_rotated[1], image_refpoint_2_rotated[1]], color='red',
                marker='x')
        plt.xlabel('x coord [pix]')
        plt.ylabel('y coord [pix]')
        plt.title('Rotated image')
        plt.show()

    return(x_array_profile, profile_left, profile_right, profile_middle, profile_left_smooth, profile_right_smooth)

def image_analyser(map_location, scale_pix, grav, density, surftens, masterplot, switch_2ndwav_param=50):
    """READING FILES"""
    print(map_location)
    valid_images = [".jpg",".JPG"]

    image_list = []

    for f in os.listdir(map_location):
        ext = os.path.splitext(f)[1]

        if ext.lower() not in valid_images:
            continue
        image = Image.open(os.path.join(map_location,f))
        image_list.append(image)


    image_list_rotated_bw, image_list_angle = [], []
    list_refpoint_1, list_refpoint_2 = [], []
    list_rodtop = []
    list_stream_left, list_stream_right = [], []
    list_stream_left_smooth, list_stream_right_smooth = [], []
    list_stream_middle = []
    wavelengths_measured_left_1st, wavelengths_measured_left_2nd = [], []
    wavelengths_measured_right_1st, wavelengths_measured_right_2nd = [], []

    radii_mean_left_1st, radii_mean_left_2nd = [], []
    radii_mean_right_1st, radii_mean_right_2nd = [], []

    switchto2ndwav = False

    for n in range(len(image_list)):
        middle = np.array([np.shape(image_list[n])[1] / 2, np.shape(image_list[n])[0] / 2])

        image_rotated, image_angle, image_refpoint_1, image_refpoint_2, image_rodtop \
            = image_rotator(image_list[n], plot=True)

        image_rotated_bw = np.sum(np.array(image_rotated), axis=2)
        image_list_rotated_bw.append(image_rotated_bw)
        image_list_angle.append(image_angle)
        list_refpoint_1.append(image_refpoint_1)
        list_refpoint_2.append(image_refpoint_2)
        list_rodtop.append(image_rodtop)

        image_array_profile, image_profile_left, image_profile_right, image_profile_middle, image_profile_left_smooth\
        , image_profile_right_smooth = edge_finder(image_rotated_bw, image_refpoint_1, image_refpoint_2\
                                                   , image_angle, middle, plot=True, extensiveplot=True)

        streamlen_approx = (355 + image_refpoint_2[0] - image_refpoint_1[0])/scale_pix
        wavelen_approx = np.pi*surftens/(density*grav*streamlen_approx)
        wavelen_pix_approx = wavelen_approx*scale_pix
        peaksearch_order = min(20, int(wavelen_pix_approx/4))

        troughs_left = np.array(sc.argrelmax(image_profile_left_smooth, order=peaksearch_order))[0]
        troughs_right = np.array(sc.argrelmax(-image_profile_right_smooth, order=peaksearch_order))[0]

        wavelengths_pix_left = np.diff(troughs_left)
        wavelengths_pix_right = np.diff(troughs_right)



        image_wavelength_measured_left_1st = wavelengths_pix_left[0] / scale_pix
        wave_profile_left_1st = image_profile_left[troughs_left[0]:troughs_left[1]] \
                                - image_profile_middle[troughs_left[0]:troughs_left[1]]
        image_wavelength_measured_right_1st = wavelengths_pix_right[0] / scale_pix
        wave_profile_right_1st = image_profile_right[troughs_right[0]:troughs_right[1]] \
                                 - image_profile_middle[troughs_right[0]:troughs_right[1]]

        wave_meanradius_left_1st = np.mean(wave_profile_left_1st)
        wave_meanradius_right_1st = np.mean(wave_profile_right_1st)

        try:
            image_wavelength_measured_left_2nd = wavelengths_pix_left[1]/scale_pix
            wave_profile_left_2nd = image_profile_left[troughs_left[1]:troughs_left[2]] \
                                    - image_profile_middle[troughs_left[1]:troughs_left[2]]
            image_wavelength_measured_right_2nd = wavelengths_pix_right[1]/scale_pix
            wave_profile_right_2nd = image_profile_right[troughs_right[1]:troughs_right[2]] \
                                     - image_profile_middle[troughs_right[1]:troughs_right[2]]

            wave_meanradius_left_2nd = np.mean(wave_profile_left_2nd)
            wave_meanradius_right_2nd = np.mean(wave_profile_right_2nd)


        except:
            image_wavelength_measured_left_2nd = None
            wave_profile_left_2nd = np.array([])
            image_wavelength_measured_right_2nd = None
            wave_profile_right_2nd = np.array([])

            wave_meanradius_left_2nd = None
            wave_meanradius_right_2nd = None

        wavelengths_measured_left_1st.append(image_wavelength_measured_left_1st)
        wavelengths_measured_left_2nd.append(image_wavelength_measured_left_2nd)

        wavelengths_measured_right_1st.append(image_wavelength_measured_right_1st)
        wavelengths_measured_right_2nd.append(image_wavelength_measured_right_2nd)


        radii_mean_left_1st.append(wave_meanradius_left_1st)
        radii_mean_left_2nd.append(wave_meanradius_left_2nd)

        radii_mean_right_1st.append(wave_meanradius_right_1st)
        radii_mean_right_2nd.append(wave_meanradius_right_2nd)

        if min(troughs_left[0], troughs_right[0]) < switch_2ndwav_param:
            switchto2ndwav = True
            print('Switching to second wavelength')


        if masterplot:
            plt.figure(figsize=(15, 10))
            plt.plot(image_array_profile, image_profile_left_smooth, color='red', linewidth=3)
            plt.plot(image_array_profile, image_profile_right_smooth, color='red', linewidth=3)
            plt.plot(image_array_profile, image_profile_left, color='blue')
            plt.plot(image_array_profile, image_profile_right, color='blue')
            plt.scatter(image_array_profile[troughs_left], image_profile_left_smooth[troughs_left], color='black')
            plt.scatter(image_array_profile[troughs_right], image_profile_right_smooth[troughs_right], color='black')
            #plt.xlim(2200, 2300)
            #plt.ylim(1600, 2200)
            plt.show()

            print('left wavelength = ' + str(image_wavelength_measured_left_1st) \
                  + " " + str(image_wavelength_measured_left_2nd))
            print('right wavelength = ' + str(image_wavelength_measured_right_1st) \
                  + " " + str(image_wavelength_measured_right_2nd))

    if switchto2ndwav:
        print('Switch has happened.')

    if switchto2ndwav==False:
        radii_left = np.abs(radii_mean_left_1st)/scale_pix
        radii_right = np.abs(radii_mean_right_1st)/scale_pix
        wavelengths_left = wavelengths_measured_left_1st
        wavelengths_right = wavelengths_measured_right_1st

    else:
        try:
            radii_left = np.abs(radii_mean_left_2nd)/scale_pix
            radii_right = np.abs(radii_mean_right_2nd)/scale_pix
            wavelengths_left = wavelengths_measured_left_2nd
            wavelengths_right = wavelengths_measured_right_2nd
        except TypeError:
            print("Something went wrong. It might be the case that switch was made to 2nd wave "\
                  "but some image only had 1 wave available.")
            exit()

    radius_left = np.mean(radii_left)
    radius_left_error = np.std(radii_left)/np.sqrt(len(radii_left))
    radius_right = np.mean(radii_right)
    radius_right_error = np.std(radii_right)/np.sqrt(len(radii_right))
    radii_master = np.append(radii_left, radii_right)
    radius = np.mean(radii_master)
    radius_error = np.std(radii_master)/np.sqrt(len(radii_master))
    radius_collection = np.array([radius_left, radius_left_error, radius_right, radius_right_error, radius, radius_error])

    wavelen_left = np.mean(wavelengths_left)
    wavelen_left_error = np.std(wavelengths_left)/np.sqrt(len(wavelengths_left))
    wavelen_right = np.mean(wavelengths_right)
    wavelen_right_error = np.std(wavelengths_right)/np.sqrt(len(wavelengths_right))
    wavelengths_master = np.append(wavelengths_left, wavelengths_right)
    wavelen = np.mean(wavelengths_master)
    wavelen_error = np.std(wavelengths_master)/np.sqrt(len(wavelengths_master))

    wavelen_collection = np.array([wavelen_left, wavelen_left_error, wavelen_right, wavelen_right_error, wavelen, wavelen_error])

    return(radius_collection, wavelen_collection, switchto2ndwav)

# Constants are defined:
g = 9.81  # 9.812
g_error = 0.01  # 0.001
water_d = 997
water_d_error = 1
water_surft = 72.8e-3
water_surft_error = 0.1e-3
diameter_nozzle = 6e-3
diameter_nozzle_error = 5e-5

# Stream flow rate is calculated from measurements:
volume_filled = 4e-4
volume_filled_error = 1e-6
time_filled_list = np.loadtxt("Flowrate_Measurement_Newdata.txt", delimiter=",", dtype=str).astype(float)
time_filled_N = len(time_filled_list) * len(time_filled_list[0])
time_filled = np.mean(time_filled_list)
time_filled_error = np.std(time_filled_list) / np.sqrt(time_filled_N)
print('Filltime ', time_filled, time_filled_error)

flow_rate = volume_filled / (20 * time_filled)
flow_rate_error = flow_rate * (
            (((volume_filled_error / volume_filled) ** 2) + ((time_filled_error / 20 * time_filled) ** 2)) ** 0.5)
print('flow rate', flow_rate, flow_rate_error)

cal_pixnum_list = np.array([4550, 4547, 4546, 4535])
cal_pixnum = np.mean(cal_pixnum_list)
cal_pixnum_error = np.std(cal_pixnum_list) / np.sqrt(len(cal_pixnum_list))
cal_length = 69.72 * 1e-3 #meter
cal_length_error = 0.01 * 1e-3
scale = cal_pixnum / cal_length
scale_error = scale * ((((cal_pixnum_error / cal_pixnum) ** 2) + ((cal_length_error / cal_length) ** 2)) ** 0.5)
print('scale', scale, scale_error)

# Wavelengths and average radii are extracted from images
heights = 2
stream_radii = np.zeros(heights)
stream_radii_error = np.zeros(heights)
stream_wavelen = np.zeros(heights)
stream_wavelen_error = np.zeros(heights)
switches = np.zeros(heights)

for heightnum in range(heights):
    path = r'C:\Users\woutg\Wavelength Measurements\Height ' + str(heightnum + 1)

    radius_collection_test, wavelen_collection_test, wave_switch = image_analyser(path, scale, g, water_d, water_surft\
                                                                            , masterplot=True, switch_2ndwav_param=45)
    print(radius_collection_test, wavelen_collection_test, wave_switch)

    stream_radii[heightnum], stream_radii_error[heightnum] = radius_collection_test[4], radius_collection_test[5]
    stream_wavelen[heightnum], stream_wavelen_error[heightnum] = wavelen_collection_test[4], wavelen_collection_test[5]
    switches[heightnum] = wave_switch

stream_velocity = flow_rate/(np.pi*(stream_radii**2))
stream_velocity_error = np.sqrt((flow_rate_error/(np.pi*stream_radii**2))**2 \
                                + (2*flow_rate*stream_radii_error/(np.pi*stream_radii**3))**2)



plt.errorbar(stream_velocity, 1e3*stream_wavelen, xerr=stream_velocity_error, yerr=1e3*stream_wavelen_error)
plt.xlabel('Stream velocity [m/s]')
plt.ylabel('Wavelength [mm]')
plt.title('Dispersion Relation')
plt.show()

print('Switches:' + str(switches))

