import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import ipdb
import sklearn.cluster
from tqdm import trange

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''


    height = img.shape[0]
    width = img.shape[1]
    
    if img.ndim < 3:
        # Must be grayscale. Generate three repeated channels to make 3-channel
        img = np.tile(img[:,:,np.newaxis],(1,1,3))

    depth = img.shape[2]
    img_temp = np.copy(img)
    img_temp = img_temp.reshape(width*height*depth)

    # if all(img_temp <= 1.0) and all(img_temp >= 0):
    #     # Do nothing
    #     print('Satisfied')
    # else:
    #     img = np.array(img).astype(np.float32)/255

    img = np.array(img).astype(np.float32)/255

    img = skimage.color.rgb2lab(img)

    # M_gauss = np.empty((height,width,3*len(opts.filter_scales)),dtype=float)
    # M_Laplace = np.empty((height,width,3*len(opts.filter_scales)),dtype=float)
    # M_gauss_der_x = np.empty((height,width,3*len(opts.filter_scales)),dtype=float)
    # M_gauss_der_y = np.empty((height,width,3*len(opts.filter_scales)),dtype=float)

    M = np.empty((height,width,3*4*len(opts.filter_scales)), dtype=float)

    for i in range(len(opts.filter_scales)):
        
        gaussian_response = np.empty(np.shape(img),dtype=float)

        M[:,:,i*12 + 0] = scipy.ndimage.gaussian_filter(img[:,:,0], opts.filter_scales[i])
        M[:,:,i*12 + 1] = scipy.ndimage.gaussian_filter(img[:,:,1], opts.filter_scales[i])
        M[:,:,i*12 + 2] = scipy.ndimage.gaussian_filter(img[:,:,2], opts.filter_scales[i])

        M[:,:,i*12 + 3] = scipy.ndimage.gaussian_laplace(img[:,:,0],opts.filter_scales[i])
        M[:,:,i*12 + 4] = scipy.ndimage.gaussian_laplace(img[:,:,1],opts.filter_scales[i])
        M[:,:,i*12 + 5] = scipy.ndimage.gaussian_laplace(img[:,:,2],opts.filter_scales[i])

        M[:,:,i*12 + 6] = scipy.ndimage.gaussian_filter(img[:,:,0],opts.filter_scales[i],(1,0))
        M[:,:,i*12 + 7] = scipy.ndimage.gaussian_filter(img[:,:,1],opts.filter_scales[i],(1,0))
        M[:,:,i*12 + 8] = scipy.ndimage.gaussian_filter(img[:,:,2],opts.filter_scales[i],(1,0))

        M[:,:,i*12 + 9] = scipy.ndimage.gaussian_filter(img[:,:,0],opts.filter_scales[i], (0,1))
        M[:,:,i*12 + 10] = scipy.ndimage.gaussian_filter(img[:,:,1],opts.filter_scales[i], (0,1))
        M[:,:,i*12 + 11] = scipy.ndimage.gaussian_filter(img[:,:,2],opts.filter_scales[i], (0,1))


    filter_responses = M

    return filter_responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----

    # 1. Read an image

    # 2. Extract responses

    # 3. Save to temporary file

    pass

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    alpha = opts.alpha
    filter_scales = opts.filter_scales
    num_of_filters = 4
    K = opts.K

    # 1. Load the training dada
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    T = len(train_files) # Number of training images
    F = num_of_filters*len(opts.filter_scales) # Size of filter bank

    filter_responses = np.empty((alpha*T,3*F),dtype=float) # Initialize matrix for filter responses

    # 2. Extract alpha*T filter responses over the training files
    for i in trange(T):

        # Call compute_dictionary_one_image()
        img_path = join(opts.data_dir,train_files[i])
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)
        img_shape = img.shape
        # img_lab = skimage.color.rgb2lab(img)
        filter_response_one_image = extract_filter_responses(opts, img)

        height = filter_response_one_image.shape[0]
        width = filter_response_one_image.shape[1]
        sample = np.random.choice(height*width, alpha, replace = False)
        filter_response_sampled = np.squeeze(filter_response_one_image[(sample/width).astype(int), (sample%width).astype(int), :])

        filter_responses[alpha*i:alpha*i + alpha,:] = filter_response_sampled

    # 3. Call K-means

    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_

    np.save(join(out_dir, 'dictionary.npy'), dictionary)

    pass

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.
    
    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    # img_lab = skimage.color.rgb2lab(img)
    filter_responses = extract_filter_responses(opts, img)
    filter_height = filter_responses.shape[0]
    filter_width = filter_responses.shape[1]
    filter_depth = filter_responses.shape[2]
    
    filter_responses_distance = filter_responses.reshape((filter_height*filter_width,filter_depth))

    distance_matrix = scipy.spatial.distance.cdist(filter_responses_distance, dictionary, metric='euclidean')
    cluster_assignment = np.argmin(distance_matrix,axis=1)
    wordmap = cluster_assignment.reshape((filter_height,filter_width))

    return wordmap

