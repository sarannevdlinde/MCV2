import cv2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import poisson_editing
import poisson_editing2

# Load images
src = cv2.imread('images/lena/girl.png')
dst = cv2.imread('images/lena/lena.png')
# For Mona Lisa and Ginevra:
# src = cv2.imread('images/monalisa/ginevra.png')
# dst = cv2.imread('images/monalisa/monalisa.png')

# Customize the code with your own pictures and masks.

# Store shapes and number of channels (src, dst and mask should have same dimensions!)
ni, nj, nChannels = dst.shape

# Load masks for eye swapping
src_mask_eyes = cv2.imread('images/lena/mask_src_eyes.png', cv2.IMREAD_COLOR)
dst_mask_eyes = cv2.imread('images/lena/mask_dst_eyes.png', cv2.IMREAD_COLOR)

# Load masks for mouth swapping
src_mask_mouth = cv2.imread('images/lena/mask_src_mouth.png', cv2.IMREAD_COLOR)
dst_mask_mouth = cv2.imread('images/lena/mask_dst_mouth.png', cv2.IMREAD_COLOR)

# Get the translation vectors (hard coded)
t_eyes = poisson_editing.get_translation(src_mask_eyes, dst_mask_eyes, "eyes")
t_mouth = poisson_editing.get_translation(src_mask_mouth, dst_mask_mouth, "mouth")

eyes_src = cv2.bitwise_and(src, src, mask=cv2.cvtColor(src_mask_eyes, cv2.COLOR_BGR2GRAY))
eyes_dst = np.zeros_like(eyes_src)
eyes_dst[dst_mask_eyes[:, :, 0] > 0] = eyes_src[src_mask_eyes[:, :, 0] > 0]

mouth_src = cv2.bitwise_and(src, src, mask=cv2.cvtColor(src_mask_mouth, cv2.COLOR_BGR2GRAY))
mouth_dst = np.zeros_like(mouth_src)
mouth_dst[dst_mask_mouth[:, :, 0] > 0] = mouth_src[src_mask_mouth[:, :, 0] > 0]

#Combine the two shifted parts into the combined image
mask = dst_mask_eyes + dst_mask_mouth

cv2.imshow('mask', mask)
cv2.waitKey(0)
u_comb = dst.copy()
u_comb[dst_mask_eyes[:, :, 0] > 0] = eyes_dst[dst_mask_eyes[:, :, 0] > 0]
u_comb[dst_mask_mouth[:, :, 0] > 0] = mouth_dst[dst_mask_mouth[:, :, 0] > 0]
cv2.imshow('ucomb', u_comb)
cv2.waitKey(0)

def E(u, B, b, c, M, N):
    u = np.reshape(u, (M, N))
    A_u = poisson_editing.poisson_linear_operator(u, B)
    print("minimizing")
    return (0.5 * np.dot(A_u, u.flatten()) - np.dot(b, u.flatten()) + c)

for channel in range(3):
    m = mask[:, :, channel]
    u = u_comb[:, :, channel]
    u2 = dst[:, :, channel]
    u1 = src[:, :, channel]

    m_flat = m.flatten()
    u_flat = u.flatten()
    u2_flat = u2.flatten()
    u1_flat = u1.flatten()

    beta_0 = 1  # TRY CHANGING
    beta = (beta_0 * (1 - m)).flatten()
    B = np.diag(beta)

    vi, vj = poisson_editing.composite_gradients(u1, u2, m)
    b = np.dot(B, u2_flat) - poisson_editing.im_bwd_divergence(vi, vj)

    v = np.vstack((vi, vj))

    v_inner = np.sum(vi * vi) + np.sum(vj * vj)
    x = np.dot(B, u2_flat)
    c = 0.5 * v_inner + 0.5 * np.dot(x, u2_flat.T)

    M, N = u.shape
    result = minimize(E, u_flat, args=(B, b, c, M, N), method='L-BFGS-B', options={"maxiter": 2})

    optimal_u = result.x
    print("u that minimizes E(u): ", optimal_u)
    print("min value for E(u): ", result.fun)

cv2.imshow('Final result of Poisson blending', u_final)
