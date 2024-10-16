import numpy as np
from scipy.signal import correlate2d
from scipy.sparse import diags

def im_fwd_gradient(image: np.ndarray):

    # Calculate forward gradients
    grad_i = np.zeros_like(image)
    grad_j = np.zeros_like(image)

    # Forward gradient in x-direction (horizontal)
    grad_i[:, :-1] = image[:, 1:] - image[:, :-1]

    # Forward gradient in y-direction (vertical)
    grad_j[:-1, :] = image[1:, :] - image[:-1, :]

    # Stack gradients into a single array
    return np.stack((grad_i, grad_j), axis=0)  # Shape (2, M, N)

def compute_composite_gradient(u1, u2, m):
    """
    Computes the composite gradient vector v using the binary mask m.

    Parameters:
    u1 (np.ndarray): First input image as a 2D array (shape: (M, N)).
    u2 (np.ndarray): Second input image as a 2D array (shape: (M, N)).
    m (np.ndarray): Binary mask as a 2D array (shape: (M, N)).

    Returns:
    np.ndarray: Composite gradient vector v of shape (2, M, N).
    """
    # Compute forward gradients for both images
    grad_u1 = im_fwd_gradient(u1)  # Shape (2, M, N)
    grad_u2 = im_fwd_gradient(u2)  # Shape (2, M, N)

    # Extract horizontal and vertical gradients
    grad_i_u1, grad_j_u1 = grad_u1  # Shapes: (M, N), (M, N)
    grad_i_u2, grad_j_u2 = grad_u2  # Shapes: (M, N), (M, N)

    # Initialize the composite gradient vector
    v_i = m * grad_i_u1 + (1 - m) * grad_i_u2  # Horizontal component (shape: (M, N))
    v_j = m * grad_j_u1 + (1 - m) * grad_j_u2  # Vertical component (shape: (M, N))

    # Stack the composite gradients into a single array
    v = np.stack((v_i, v_j), axis=0)  # Shape (2, M, N)

    return v

def im_bwd_divergence(vi, vj):
    M, N = vi.shape
    div = np.zeros((M, N), dtype=np.float32)

    # Use central differences for divergence calculation
    div[:-1, :] += vi[1:, :] - vi[:-1, :]  # ∂v_x/∂y
    div[:, :-1] += vj[:, 1:] - vj[:, :-1]  # ∂v_y/∂x

    return div  # Keep original values for boundary

def poisson_linear_operator(u, beta):
    M, N = u.shape
    MN = M * N
    
    # Create the diagonals for the Laplacian
    diagonals = [-4 * np.ones(MN), 
                 np.ones(MN - 1), 
                 np.ones(MN - 1), 
                 np.ones(MN - N), 
                 np.ones(MN - N)]
    
    # Adjust for boundaries without removing connections
    # For horizontal boundaries (first and last rows)
    for j in range(N):  # Iterate over columns
        # For upper boundary (first row)
        if j > 0:  # No left connection for the first row
            diagonals[1][j] = 0  # No left connection
        # For lower boundary (last row)
        if j < N - 1:  # No right connection for the last row
            diagonals[2][(M - 1) * N + j] = 0  # No right connection

    # Create the sparse matrix
    A = diags(diagonals, [0, -1, 1, -N, N], shape=(MN, MN), format='csr')
    
    # Add the beta coefficients
    A += diags(beta.flatten(), 0)

    return A

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