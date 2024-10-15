import cv2
import numpy as np
from scipy.optimize import minimize

import poisson_editing

# Load images
src = cv2.imread('images/lena/girl.png')
dst = cv2.imread('images/lena/lena.png')
# For Mona Lisa and Ginevra:
# src = cv2.imread('images/monalisa/ginevra.png')
# dst = cv2.imread('images/monalisa/monalisa.png')

# Customize the code with your own pictures and masks.

# Store shapes and number of channels (src, dst and mask should have same dimensions!)
ni, nj, nChannels = dst.shape

# Display the images
cv2.imshow('Source image', src);
cv2.waitKey(0)
cv2.imshow('Destination image', dst);
cv2.waitKey(0)

# Load masks for eye swapping
src_mask_eyes = cv2.imread('images/lena/mask_src_eyes.png', cv2.IMREAD_COLOR)
dst_mask_eyes = cv2.imread('images/lena/mask_dst_eyes.png', cv2.IMREAD_COLOR)
cv2.imshow('Eyes source mask', src_mask_eyes);
cv2.waitKey(0)
cv2.imshow('Eyes destination mask', dst_mask_eyes);
cv2.waitKey(0)

# Load masks for mouth swapping
src_mask_mouth = cv2.imread('images/lena/mask_src_mouth.png', cv2.IMREAD_COLOR)
dst_mask_mouth = cv2.imread('images/lena/mask_dst_mouth.png', cv2.IMREAD_COLOR)
cv2.imshow('Mouth source mask', src_mask_mouth);
cv2.waitKey(0)
cv2.imshow('Mouth destination mask', dst_mask_mouth);
cv2.waitKey(0)

# Get the translation vectors (hard coded)
t_eyes = poisson_editing.get_translation(src_mask_eyes, dst_mask_eyes, "eyes")
t_mouth = poisson_editing.get_translation(src_mask_mouth, dst_mask_mouth, "mouth")

# Cut out the relevant parts from the source image and shift them into the right position
# CODE TO COMPLETE

# Blend with the original (destination) image
# CODE TO COMPLETE

mask = np.zeros_like(dst)
u_comb = np.zeros_like(dst)  # combined image

def E(u):
    u = np.array(u) 
    return 0.5 * np.dot(u.T, np.dot(A, u)) - np.dot(b, u) + c

for channel in range(3):
    m = mask[:, :, channel]
    u = u_comb[:, :, channel]
    u2 = dst[:, :, channel]
    u1 = src[:, :, channel]

    beta_0 = 1  # TRY CHANGING
    beta = beta_0 * (1 - mask)

    vi, vj = poisson_editing.composite_gradients(u1, u2, mask)
    A = poisson_editing.poisson_linear_operator(u, beta)
    b = (beta * u2) - poisson_editing.im_bwd_divergence(vi, vj)
    v = np.array([vi, vj])
    c = 0.5 * (np.inner(v, v)) + 0.5 * (np.inner((beta * u2), u2))

    initial_u = np.zeros(A.shape[1])

    result = minimize(E, initial_u)

    optimal_u = result.x
    print("u that minimizes E(u): ", optimal_u)
    print("min value for E(u): ", result.fun)
    u_final =

cv2.imshow('Final result of Poisson blending', u_final)
