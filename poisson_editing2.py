import numpy as np
from scipy.signal import correlate2d

def im_fwd_gradient(image: np.ndarray):

    # Calculate forward gradients
    grad_i = np.zeros_like(image)
    grad_j = np.zeros_like(image)

    # Forward gradient in x-direction (horizontal)
    grad_i[:, :-1] = image[:, 1:] - image[:, :-1]

    # Forward gradient in y-direction (vertical)
    grad_j[:-1, :] = image[1:, :] - image[:-1, :]
    return grad_i, grad_j

def im_bwd_divergence(vi: np.ndarray, vj: np.ndarray):

    # CODE TO COMPLETE

    # Calculate divergence
    div_i = np.zeros_like(vi, dtype=np.float32)
    div_j = np.zeros_like(vj, dtype=np.float32)

    # Divergence: ∂v_x/∂x + ∂v_y/∂y
    div_i[:-1, :] = vi[1:, :] - vi[:-1, :]  # ∂v_x/∂y
    div_j[:, :-1] = vj[:, 1:] - vj[:, :-1]  # ∂v_y/∂x

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
       
    # Compute gradients of im1
    grad1_x,grad1_y = im_fwd_gradient(u1)

    # Compute gradients of im2
    grad2_x,grad2_y = im_fwd_gradient(u2)
    # CODE TO COMPLETE
        # Initialize composite gradients
    vi = np.zeros_like(grad1_x)
    vj = np.zeros_like(grad1_y)

    # Combine gradients based on the mask
    vi[mask >= 1] = grad1_x[mask >= 1]
    vj[mask >= 1] = grad1_y[mask >= 1]

    vi[mask == 0] = grad2_x[mask == 0]
    vj[mask == 0] = grad2_y[mask == 0]
    
    return vi, vj

def poisson_linear_operator(u: np.array, beta: np.array):
    """
    Implements the action of the matrix A in the quadratic energy associated
    to the Poisson editing problem.
    """
        # Get the dimensions of the input array
    height, width = u.shape
    
    # Initialize the output array
    Au = np.zeros_like(u)

    # Loop through each pixel in the interior of the image
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Discrete Laplacian using central differences
            Laplacian = (
                u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4 * u[i, j]
            )
            Au[i, j] = beta[i, j] * Laplacian
    
    # Handle boundary conditions (you may want to set them to zero or some other value)
    Au[0, :] = 0
    Au[-1, :] = 0
    Au[:, 0] = 0
    Au[:, -1] = 0

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