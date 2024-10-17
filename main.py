import cv2
import numpy as np
from poisson_editing import *
from scipy.optimize import minimize

# Load images
#src = cv2.imread('images/lena/girl.png')
#dst = cv2.imread('images/lena/lena.png')
# For Mona Lisa and Ginevra:
# src = cv2.imread('images/monalisa/ginevra.png')
# dst = cv2.imread('images/monalisa/monalisa.png')

# Customize the code with your own pictures and masks.
src = cv2.imread('C:/Users/laila/Downloads/CV2425_project_week22/CV2425_project_week2/images/monalisa/ginevra.png',cv2.IMREAD_COLOR)
dst = cv2.imread('C:/Users/laila/Downloads/CV2425_project_week22/CV2425_project_week2/images/monalisa/lisa.png',cv2.IMREAD_COLOR)
original_height, original_width = src.shape[:2]
src = cv2.resize(src, (256, 256))
dst = cv2.resize(dst, (256, 256))
# Store shapes and number of channels (src, dst and mask should have same dimensions!)
ni, nj, nChannels = dst.shape

# Display the images
cv2.imshow('Source image', src); cv2.waitKey(0)
cv2.imshow('Destination image', dst); cv2.waitKey(0)

# Load masks for eye swapping
#src_mask_eyes = cv2.imread('images/lena/mask_src_eyes.png', cv2.IMREAD_COLOR)
#dst_mask_eyes = cv2.imread('images/lena/mask_dst_eyes.png', cv2.IMREAD_COLOR)
#cv2.imshow('Eyes source mask', src_mask_eyes); cv2.waitKey(0)
#cv2.imshow('Eyes destination mask', dst_mask_eyes); cv2.waitKey(0)
original_mask = cv2.imread('C:/Users/laila/Downloads/CV2425_project_week22/CV2425_project_week2/images/monalisa/mask.png',cv2.IMREAD_GRAYSCALE)  # Replace with actual original mask
translated_mask = cv2.imread('C:/Users/laila/Downloads/CV2425_project_week22/CV2425_project_week2/images/monalisa/mask.png',cv2.IMREAD_GRAYSCALE)   # Replace with actual translated mask
original_mask = cv2.resize(original_mask, (256, 256))
translated_mask = cv2.resize(translated_mask, (256, 256))
#offset = get_translation(original_mask, translated_mask)
converted_mask = np.where(translated_mask >= 1, 1, translated_mask) # ensure mask is zeros and 1s
# Load masks for mouth swapping
#src_mask_mouth = cv2.imread('images/lena/mask_src_mouth.png', cv2.IMREAD_COLOR)
#dst_mask_mouth = cv2.imread('images/lena/mask_dst_mouth.png', cv2.IMREAD_COLOR)
#cv2.imshow('Mouth source mask', src_mask_mouth); cv2.waitKey(0)
#cv2.imshow('Mouth destination mask', dst_mask_mouth); cv2.waitKey(0)

# Get the translation vectors (hard coded)
#t_eyes = get_translation(src_mask_eyes, dst_mask_eyes, "eyes")
#t_mouth = get_translation(src_mask_mouth, dst_mask_mouth, "mouth")

# Cut out the relevant parts from the source image and shift them into the right position
#y_max, x_max = dst.shape[:-1]
#y_min, x_min = 0, 0
#x_range = x_max - x_min
#y_range = y_max - y_min
#offset=(24,8)
#M = np.float32([[1, 0, offset[1]], [0, 1, offset[0]]])
#src = cv2.warpAffine(src, M, (x_range, y_range))
translated_image=src
#print(translated_image.shape)
# Define the cropping box (y1:y2, x1:x2)
crop_box = translated_image#[100:300, 100:300]  # Adjust these values as needed

#####################################################################################
#Crop the target image too 
# Define the cropping box (y1:y2, x1:x2)
crop_box_dst = dst#[100:300, 100:300] # Adjust these values as needed

#################################################################################
#Crop the mask 
crop_box_mask = converted_mask#[100:300, 100:300]  # Adjust these values as needed

# Blend with the original (destination) image
combined_image = np.zeros_like(crop_box_dst) # combined image

for channel in range(3):
    # Initialize combined image
    combined_image[:, :, channel] = converted_mask * crop_box[:, :, channel] + (1 - converted_mask) * crop_box_dst[:, :, channel]
    m = converted_mask
    u = combined_image[:, :, channel]
    f = crop_box_dst[:, :, channel]
    u1 = crop_box[:, :, channel]
    source_flat = u1.flatten()
    target_flat = f.flatten()
    mask_flat = m.flatten()
    
    # Calculate beta based on the mask
    beta_0 = 1 #playing around with beta affects the result's intensity 
    beta = beta_0 * (1 - m)

    # Extract dimensions
    original_u_shape = u.shape
    y_min, y_max = 0, original_u_shape[0]
    x_min, x_max = 0, original_u_shape[1]

    # Precompute gradients and divergence
    vi, vj = composite_gradients(u1, f, m)
    div = im_bwd_divergence(vi, vj)

    # Compute b vector only once
    Bu2 = beta * f
    b = Bu2 - div

    # Define the quadratic energy function
    def E(u):
        u_reshaped = u.reshape(original_u_shape)
        gradients = poisson_linear_operator(u_reshaped, beta)
        energy = (0.5 * np.dot(gradients.flatten(), u) - np.dot(b.flatten(), u) +
                0.5 * np.dot(vi.flatten(), vj.flatten()) +  
                0.5 * np.dot(Bu2.flatten(), target_flat))
        return energy


    # Perform the optimization 
    result = minimize(E, u.flatten(), method='CG', 
                    options={'maxiter': 3, 'disp': True}) #low number of iterations because it is slow 

    # Reshape the result back to original dimensions if needed
    optimized_image = result.x.reshape(original_u_shape)
    combined_image[:, :, channel] = optimized_image

final = cv2.resize(combined_image, (original_width, original_height))
cv2.imshow('Final result of Poisson blending', final)