import numpy as np

def devernay(img, sigma=0.0, th_l=0.0, th_h=0.0):
    '''
    '''
    from .cdevernay import c_devernay
    if img.dtype != np.double:
        img = np.double(img)
    return c_devernay(img, sigma, th_l, th_h)

if __name__ == '__main__':
    img = np.zeros((256, 256), dtype=np.uint8)

    for i in range(50):
        img[i+10, i+10] = 128
        img[100, i+50] = 255
    print(devernay(img))
