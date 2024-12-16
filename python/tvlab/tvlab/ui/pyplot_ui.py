'''
Copyright (C) 2023 TuringVision

matplotlib plot interface
'''

from math import ceil
from matplotlib import pyplot as plt

__all__ = ['show_images']

def show_images(image_list, text_list=None, ncols=3, figsize=(9, 9)):
    nrows = ceil(len(image_list) / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            sharex=True, sharey=True,
                            figsize=figsize)

    axs = axs.flatten()
    for i, img in enumerate(image_list):
        if text_list:
            axs[i].set_title(text_list[i])
        axs[i].imshow(img, cmap='gray', interpolation='nearest')

    for ax in axs[len(image_list):]: ax.axis('off')

    axs[0].set_xticks([])
    axs[0].set_yticks([])
    plt.tight_layout()
