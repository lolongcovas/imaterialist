import os
import json
import torch
from torchvision.transforms import ToTensor
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


class dataset(torch.utils.data.Dataset):

    def __init__(self, bpath, im_filenames, im_labels,
                 input_size=224, validation=False):
        self.bpath = bpath
        self.gt = json.load(json_filename)
        self.input_size = input_size
        self.validation = validation
        
        sometimes1 = lambda aug: iaa.Sometimes(0.2, aug)
        sometimes2 = lambda aug: iaa.Sometimes(0.4, aug)
        self.seq = iaa.Sequential([
            iaa.Crop(px=(0, 10)), # crop images from each side by 0 to 10px (randomly chosen)
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.Flipud(0.5),
            sometimes1(iaa.GaussianBlur(sigma=(0, 3.0))),  # blur images with a sigma of 0 to 3.0
            sometimes2(
                iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 90-110% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -10 to +10 percent (per axis)
                    rotate=(-45, 45), # rotate by -45 to +45 degrees
                    shear=(-5, 5), # shear by -5 to +5 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 0), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                ))
        ])

    def __getitem__(self, index):
        im = cv2.imread(os.path.join(self.bpath, self.filenames[index]), 1)
        if not self.validation:
            im = self.seq.augment_image(im)
        rows, cols = im.shape[:2]

        im = im[(rows-self.input_size)//2:
                (rows-self.input_size)//2+self.input_size,
                (cols-self.input_size)//2:
                (cols-self.input_size)//2+self.input_size, :]
        # BGR 2 RGB
        im = im[:, :, ::-1].copy()

        # im = preprocessing.contrast_enhanced(im)

        # mean = [0.3940, 0.2713, 0.1869], RGB
        # std = [0.2777, 0.1981, 0.1574]
        im = ToTensor()(im)
        for t, m, s in zip(im,
                           [0.3940, 0.2713, 0.1869],
                           [0.2777, 0.1981, 0.1574]):
            t.sub_(m).div_(s)
        return im, self.labels[index]

    def __len__(self):
        return len(self.filenames)
