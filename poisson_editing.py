import numpy as np
from scipy.signal import correlate2d
from scipy.sparse import lil_matrix

def im_fwd_gradient(image: np.ndarray):

    grad_i = np.roll(image, -1, axis=0) - image 
    grad_j = np.roll(image, -1, axis=1) - image 
    return grad_i, grad_j

def im_bwd_divergence(im1: np.ndarray, im2: np.ndarray):

    div_i = im1 - np.roll(im1, 1, axis=0)  
    div_j = im2 - np.roll(im2, 1, axis=1)  
    return div_i + div_j

def composite_gradients(u1: np.array, u2: np.array, m: np.array):

    grad_u1_i,grad_u1_j = im_fwd_gradient(u1)
    grad_u2_i,grad_u2_j = im_fwd_gradient(u2)

    vi = m * grad_u1_i + (1 - m) * grad_u2_i
    vj= m * grad_u1_j + (1 - m) * grad_u2_j
    
    return vi,vj

def poisson_linear_operator(u: np.array, beta: np.array):
    
    grad_u_i, grad_u_j = im_fwd_gradient(u)
    div = im_bwd_divergence(grad_u_i, grad_u_j)
    Au = (beta - div) * u
    
    return Au


def get_translation(original_img: np.ndarray, translated_img: np.ndarray) -> tuple:
    """
    Calculate the translation offset of a mask between the original and translated images.

    :param original_img: The original image/mask.
    :param translated_img: The translated image/mask.
    :param part: The part of the image ('eyes' or 'mouth').
    :return: A tuple representing the (dy, dx) translation offset.
    """
    
    # Calculate the cross-correlation
    correlation = correlate2d(original_img, translated_img, mode='full')
    
    # Find the index of the maximum correlation
    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Determine the translation offset
    dy = (original_img.shape[0] - 1) -y
    dx = (original_img.shape[1] - 1) -x
    
    # Return the translation vector
    return (int(dy), int(dx))
'''
    # The following shifts are hard coded:
    if part[0] == "eyes":
        return (24, 8)
    elif part[0] == "mouth":
        return (33, 7)
    else:
        return (0, 0)
'''
    # Here on could determine the shift vector programmatically,
    # given an original image/mask and its translated version.
    # Idea: using maximal cross-correlation (e.g., scipy.signal.correlate2d), or similar.