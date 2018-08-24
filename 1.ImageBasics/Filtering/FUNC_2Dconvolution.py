import numpy as np

def conv2D(img,filt1):
    ########convert to double
    img = np.float64(img)
    ###########Calculating Lengths
    nr_filt = len(filt1)    
    nc_filt = len(filt1[0])
    nr_img = len(img)
    nc_img = len(img[0])
    #######New convolved matrix dimensions
    tr = nr_img + nr_filt - 1
    tc = nc_img + nc_filt - 1
    #########Matrices used for storing the "FULL"(NEW DIMENSION) matrix  - Initialisation
    image_conv = np.zeros((tc - (nc_filt-1),tc - (nc_filt-1)))
    
    ########## Padding the image for connvolution - BOUNDARY EFFECT (zero padding mode)
    pad_image = np.pad(img,(((nr_filt-1)/2,(nc_filt-1)/2),((nr_filt-1)/2,(nc_filt-1)/2)),'constant',constant_values=(0,0))
    ######### varianble for storing the covolution sum of an INDIVIDUAL PIXEL
    sum_conv = 0

    ######## ALGO FOR 2D Convolution
    ######## ACCESSING INDIVIDUAL PIXELS of image (covers all pixels in image) Ex: 256x256=512 
    for i in range((nr_filt-1)/2,tr - (nr_filt-1)/2):
        for j in range((nc_filt-1)/2,tc - (nc_filt-1)/2):
            ########## ACCESSING INDIVIDUAL PIXELS of filter  (covers all pixels in filter) Ex: 3x3=9
            for k in range(0,nr_filt):
                for l in range(0,nc_filt):
                    ###### CALC keeping in mind the INDICES of PADED IMAGE and FILTER
                    sum_conv = sum_conv + pad_image[i-(((nr_filt-1)/2))+k, j-(((nc_filt-1)/2))+l]*filt1[k, l]
            ########## Saving output after calc of EACH PIXEL and re-init sum to zero for calc of next pixel                                        
            image_conv[i-(nr_filt-1)/2,j-(nc_filt-1)/2] = sum_conv
            sum_conv = 0                              
    ################################
    #############Normalising image to 0 to 255
    max_im = np.amax(image_conv)
    min_im = np.amin(image_conv)
    image_filt = np.round((image_conv-min_im)*255/(max_im-min_im));
    ############ Re-padding image to convert ACTUAL DIMENSIONS(512) TO NEW DIMENSIONS(514)

    image_filt1 = np.pad(image_filt,(((nr_filt-1)/2,(nc_filt-1)/2),((nr_filt-1)/2,(nc_filt-1)/2)),'constant',constant_values=(0,0))

    ############Converting back , double to unit8
    image_filt = np.uint8(image_filt1)
    #####Returning value
    return image_filt1
