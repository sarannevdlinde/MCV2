import cv2
import numpy as np
from scipy.signal import correlate2d


def im_fwd_gradient(image: np.ndarray):
    """
    Compute the forward gradient in the horizontal and vertical direction.

    :return grad[0]: the gradient in the vertical direction.
    :return grad[1]: the gradient in the horizontal direction.
    """

    grad = np.gradient(image)

    return grad[0], grad[1]


def im_bwd_divergence(im1: np.ndarray, im2: np.ndarray):
    """
    Compute the backward divergence in the horizontal and vertical direction.

    :return div_i + div_j: sum of horizontal and vertical components
    """
    div_i = np.zeros_like(im1)
    div_j = np.zeros_like(im2)

    div_i[1:, :] = im1[1:, :] - im1[:-1, :]
    div_i[0, :] = im1[0, :]

    div_j[:, 1:] = im2[:, 1:] - im2[:, :-1]
    div_j[:, 0] = im2[:, 0]
    return div_i + div_j


def composite_gradients(u1: np.array, u2: np.array, mask: np.array):
    """
    Creates a vector field v by combining the forward gradient of u1 and u2.
    For pixels where the mask is 1, the composite gradient v must coincide
    with the gradient of u1. When mask is 0, the composite gradient v must coincide
    with the gradient of u2.

    :return vi: composition of i components of gradients (vertical component)
    :return vj: composition of j components of gradients (horizontal component)
    """

    grad_u1 = im_fwd_gradient(u1)  # Get gradients of u1
    grad_u2 = im_fwd_gradient(u2)  # Get gradients of u2

    vi = np.where(mask == 1, grad_u1[0], grad_u2[0])
    vj = np.where(mask == 1, grad_u1[1], grad_u2[1])

    return vi, vj


def poisson_linear_operator(u: np.array, beta: np.array):
    """
    Implements the action of the matrix A in the quadratic energy associated
    to the Poisson editing problem.
    """
    grad = im_fwd_gradient(u)
    Au = beta - im_bwd_divergence(grad[0], grad[1])

    return Au


def get_translation(original_img: np.ndarray, translated_img: np.ndarray, *part: str):
    # For the eyes mask:
    # The top left pixel of the source mask is located at (x=115, y=101)
    # The top left pixel of the destination mask is located at (x=123, y=125)
    # This gives a translation vector of (dx=8, dy=24)

    # For the mouth mask:
    # The top left pixel of the source mask is located at (x=125, y=140)
    # The top left pixel of the destination mask is located at (x=132, y=173)
    # This gives a translation vector of (dx=7, dy=33)

    # Convert 3-dimensional img to a 2-dimensional greyscale image
    if original_img.ndim == 3:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    if translated_img.ndim == 3:
        translated_img = cv2.cvtColor(translated_img, cv2.COLOR_BGR2GRAY)

    cross_correlation = correlate2d(original_img, translated_img, mode='full')

    y_max, x_max = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
    dy = y_max - (translated_img.shape[0] - 1)
    dx = x_max - (translated_img[1] - 1)

    return dy, dx
    # The following shifts are hard coded:
    # if part[0] == "eyes":
    #     return (24, 8)
    # elif part[0] == "mouth":
    #     return (33, 7)
    # else:
    #     return (0, 0)
