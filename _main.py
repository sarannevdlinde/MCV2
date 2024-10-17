import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import _poisson_editing


def get_images(db="lena"):
    """Get images for lena's or monalisa's test

    Args:
        db (str, optional): Dataset identifier ('lena', 'monalisa'). Defaults to "lena".

    Returns:
        tuple: src image, dst image, list of masks
    """
    if (db=="lena"):
        # Define paths
        # For Lena and Girl
        input_dir = "images/lena"
        src_img = "girl.png"
        dst_img = "lena.png"
        mask_src_eyes_path = "mask_src_eyes.png"
        mask_src_mouth_path = "mask_src_mouth.png"
        mask_dst_eyes_path = "mask_dst_eyes.png"
        mask_dst_mouth_path = "mask_dst_mouth.png"

        # eyes
        src_mask_eyes = cv2.imread(os.path.join(input_dir, mask_src_eyes_path))
        dst_mask_eyes = cv2.imread(os.path.join(input_dir, mask_dst_eyes_path))
        # mouth
        src_mask_mouth = cv2.imread(os.path.join(input_dir, mask_src_mouth_path))
        dst_mask_mouth = cv2.imread(os.path.join(input_dir, mask_dst_mouth_path))

        masks = [src_mask_eyes, dst_mask_eyes, src_mask_mouth, dst_mask_mouth]

    elif (db=="monalisa"):
        # For Mona Lisa and Ginevra
        input_dir = "images/monalisa"
        src_img = "ginevra.png"
        dst_img = "monalisa.png"
        mask_path = "mask.png"

        mask = cv2.imread(os.path.join(input_dir, mask_path))
        masks = [mask]

    # Load images
    src = cv2.imread(os.path.join(input_dir, src_img))
    dst = cv2.imread(os.path.join(input_dir, dst_img))

    return src, dst, masks

def get_shifted_mask(img, msk1, msk2):
    _img = cv2.bitwise_and(img, img, mask=msk1[:,:,0])
    _out = np.zeros_like(_img)
    # _out[msk2[:,:,0] > 0] = _img[msk1[:,:,0] > 0]
    _out[msk2[:,:,0] > 0] = msk1[msk1[:,:,0] > 0]
    return _out

def get_combined_mask(src, masks):
    """Get combined mask for lena dataset (eyes + mouth),
       shifted to dst coordinates.

    Args:
        src (numpy.ndarray): src image
        masks (list): list of four masks: [m_src_eyes, m_dst_eyes, m_src_mouth, m_dst_mouth]

    Returns:
        numpy.ndarray: src masks combined and shifted to dst mask coordinates
    """
    m_src_eyes, m_dst_eyes, m_src_mouth, m_dst_mouth = masks
    # Shift src into dst coords
    src_eyes_shift = get_shifted_mask(src, m_src_eyes, m_dst_eyes)
    src_mouth_shift = get_shifted_mask(src, m_src_mouth, m_dst_mouth)
    # Combine the two shifted parts
    mask = (m_src_eyes + m_src_mouth)/255 # we know there is no overlap between mouth and eyes
    mask_shifted = (src_eyes_shift + src_mouth_shift)/255
    return mask, mask_shifted

def get_combined_img(src, dst, mask_combined, mask_combined_shift):
    img_comb = dst.copy()
    img_comb[mask_combined_shift > 0] = src[mask_combined > 0]
    return img_comb

def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Get images
    src, dst, masks = get_images("lena")
    mask_combined, mask_combined_shift = get_combined_mask(src, masks)
    cv2.imwrite(os.path.join(output_dir, 'mask_combined.png'), (mask_combined*255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, 'mask_combined_shift.png'), (mask_combined_shift*255).astype(np.uint8))
    # Combined images
    u_comb = get_combined_img(src, dst, mask_combined, mask_combined_shift)
    cv2.imwrite(os.path.join(output_dir, 'u_combined.png'), u_comb)

    # Shifted eyes and mouth
    src_shift = np.zeros_like(dst)
    src_shift[mask_combined_shift[:,:,0] > 0] = u_comb[mask_combined_shift[:,:,0] > 0]
    cv2.imwrite('src_shift.png', src_shift)

    u_opt = np.zeros_like(dst)
    for channel in range(src.shape[2]):
        print(f"Processing channel {channel}...")
        u1 = src[:,:,channel]
        u2 = dst[:,:,channel]
        u_0 = u_comb[:,:,channel]
        m = mask_combined_shift[:,:,channel]

        M, N = u_0.shape
        beta_0 = 5
        
        # Get b = Bu2 − div−v
        B = _poisson_editing.compute_B(m, beta_0)
        b = _poisson_editing.compute_b(u1, u2, m, B)

        # Compute u as solution of Au=b
        u_ch = _poisson_editing.compute_u(u_0, b, B)

        u_opt[:,:,channel] = u_ch
        
    cv2.imwrite(os.path.join(output_dir, "u_opt.png"), u_opt)

    input("press enter to exit")

    



if __name__ == '__main__':
    main()