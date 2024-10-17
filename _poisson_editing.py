import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, cg

def im_fwd_gradient(image: np.ndarray):
    """
    Compute the forward gradient in the horizontal and vertical direction.

    :return grad_x: the gradient in the horizontal direction.
    :return grad_y: the gradient in the vertical direction.
    """
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)

    # Horizontal fwd gradient (columns)
    grad_x[:,:-1] = image[:,1:] - image[:,:-1]
    grad_x[:,-1] = image[:,0] - image[:,-1]

    # Vertical fwd gradient (rows)
    grad_y[:-1,:] = image[1:,:] - image[:-1,:]
    grad_y[-1,:] = image[0,:] - image[-1,:]

    return grad_x, grad_y


def im_bwd_divergence(im1: np.ndarray, im2: np.ndarray):
    """
    Compute the backward divergence in the horizontal and vertical direction.

    :return div_i + div_j: sum of horizontal and vertical components
    """    
    grad_x = np.zeros_like(im1)
    grad_y = np.zeros_like(im2)

    # Horizontal bwd gradient (columns)
    grad_x[1:,:] = im1[1:,:] - im1[:-1,:]
    grad_x[0,:] = im1[0,:] - im1[-1,:]

    # Vertical bwd gradient (rows)
    grad_y[:,1:] = im2[:,1:] - im2[:,:-1]
    grad_y[:,0] = im2[:,0] - im2[:,-1]

    divergence = grad_x + grad_y

    return divergence

def composite_gradients(u1: np.array, u2: np.array, mask: np.array):
    """
    Creates a vector field v by combining the forward gradient of u1 and u2.
    For pixels where the mask is 1, the composite gradient v must coincide
    with the gradient of u1. When mask is 0, the composite gradient v must coincide
    with the gradient of u2.

    :return vi: composition of i components of gradients (vertical component)
    :return vj: composition of j components of gradients (horizontal component)
    """
    # Get gradients of u1 and u2
    grad_u1_j, grad_u1_i = im_fwd_gradient(u1)
    grad_u2_j, grad_u2_i = im_fwd_gradient(u2)

    # Create composition
    # vi = mask * grad_u1_i + (1 - mask) * grad_u2_i
    vi = np.where(mask > 0, grad_u1_i, grad_u2_i)
    # vj = mask * grad_u1_j + (1 - mask) * grad_u2_j
    vj = np.where(mask > 0, grad_u1_j, grad_u2_j)

    return np.array([vi, vj])

def poisson_linear_operator(u, B, M, N):
    u_r = u.reshape((M, N))
    grad_x, grad_y = im_fwd_gradient(u_r)
    Au = B.dot(u) - im_bwd_divergence(grad_x, grad_y).flatten()
    return Au

def compute_B(m, beta_0=1):
    beta = (beta_0 * (1 - m)).flatten()
    B = diags(beta)
    return B

def compute_b(u1, u2, m, B):
    """
    Define the right-hand side vector b for the equation A u = b.
    """
    # b = Bu2 − div−v
    # Compute v
    vi, vj = composite_gradients(u1, u2, m)
    # Compute b
    b = B.dot(u2.flatten()) - im_bwd_divergence(vi, vj).flatten()

    return b

def compute_u(u_init, b, B, max_iter=10000, tol=1e-5):
    M, N = u_init.shape
    # Compute A
    def A_u(u):
        return poisson_linear_operator(u, B, M, N)

    A_u_linearOp = LinearOperator(matvec=A_u, dtype=b.dtype, shape=(b.size, b.size))

    u, info = cg(A_u_linearOp, b, x0=u_init.flatten(), rtol=tol, maxiter=max_iter)
    
    if info > 0:
        print(f"The CG method did not converge after {info} iterations.")
    elif info < 0:
        print(f"Internal error in the CG method, error code: {info}")

    return u.reshape((M,N))

