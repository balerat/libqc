import numpy as np
def rotate(x, y, theta):
    xx = x
    x = x * np.cos(theta) - y * np.sin(theta)
    y = xx * np.sin(theta) + y * np.cos(theta)
    return x, y


def center_crop(img, crop_size):
    """
    img: numpy array (H, W, C) or (H, W)
    crop_size: tuple (desired_height, desired_width)
    """
    h, w = img.shape[0], img.shape[1]
    new_h, new_w = crop_size, crop_size

    start_y = h // 2 - new_h // 2
    start_x = w // 2 - new_w // 2
    
    return img[start_y:start_y+new_h, start_x:start_x+new_w]


def stretch_coord(x, y, shx, shy, offx, offy, theta):
    """Apply the shear/offset/rotation 'stretch' transform to coordinates."""
    m00, m01, m02, m10, m11, m12 = create_transform_matrix_potential(shx, shy, offx, offy, theta)
    x_t = m00 * x + m01 * y + m02
    y_t = m10 * x + m11 * y + m12
    return x_t, y_t


def create_transform_matrix_potential(sh_x, sh_y, off_x, off_y, angle):
    c = np.cos(angle)
    s = np.sin(angle)
 
    t00 = c + sh_x * s
    t01 = -s + sh_x * c
    t10 = sh_y * c + s
    t11 = -sh_y * s + c

    m00 = c * t00 + s * t10
    m01 = c * t01 + s * t11
    m10 = -s * t00 + c * t10
    m11 = -s * t01 + c * t11

    m02 = off_x - (m00 * off_x + m01 * off_y)
    m12 = off_y - (m10 * off_x + m11 * off_y)
    
    return m00, m01, m02, m10, m11, m12

def affine_transform(img, matrix):
    h, w = img.shape[0], img.shape[1]
    out = np.zeros((h, w), dtype=np.float64)
    
    m00 = matrix[0, 0]
    m01 = matrix[0, 1]
    m02 = matrix[0, 2]
    
    m10 = matrix[1, 0]
    m11 = matrix[1, 1]
    m12 = matrix[1, 2]

    for i in range(h):  
        for j in range(w):   
            
            src_i = m00 * i + m01 * j + m02
            src_j = m10 * i + m11 * j + m12
            
            out[i, j] = bilinear_interpolation(img, src_j, src_i, h, w)

    return out

def bilinear_interpolation(img, x, y, h, w):
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
            return 0.0
    x0 = int(x)
    y0 = int(y)
    dx = x - x0
    dy = y - y0
    x1 = x0 + 1
    y1 = y0 + 1

    return (1-dx)*(1-dy)*img[y0, x0] + dx*(1-dy)*img[y0, x1] + \
           (1-dx)*dy*img[y1, x0] + dx*dy*img[y1, x1]

