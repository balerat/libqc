import numpy as np 

def generate_xy_list(img):
    x_list = np.arange(img.shape[1]) - img.shape[1] * 0.5
    y_list = np.arange(img.shape[0]) - img.shape[0] * 0.5
    x_list, y_list = np.meshgrid(x_list, y_list)
    return x_list, y_list

def generate_xy_list_scaled(img, scaling=10):
    x_list = np.arange(img.shape[1]) - img.shape[1] * 0.5
    y_list = np.arange(img.shape[0]) - img.shape[0] * 0.5
    x_list = np.linspace(x_list.min(), x_list.max(), len(x_list) * scaling)
    y_list = np.linspace(y_list.min(), y_list.max(), len(y_list) * scaling)
    dx=np.diff(x_list).mean()
    dy=np.diff(y_list).mean()
    x_list, y_list = np.meshgrid(x_list, y_list)
    return x_list, y_list, dx, dy

