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
        self.im_filenames = im_filenames
        self.im_labels = im_labels
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
        im = cv2.imread(os.path.join(self.bpath,
                                     self.im_filenames[index]), 1)
        if im is None:
            im = cv2.imread(os.path.join(self.bpath,
                                         self.im_filenames[index])+'.jpg',
                            1)
            if im is None:
                fs = os.path.join(self.bpath, self.im_filenames[index])
                os.system('echo %s >> failed_images' % fs)
                im = np.zeros([300, 300, 3], dtype=np.uint8)
        if not self.validation:
            im = self.seq.augment_image(im)
        rows, cols = im.shape[:2]
        if rows < self.input_size\
           or cols < self.input_size:
            im = cv2.resize(im, (self.input_size, self.input_size))
        else:
            # random crop
            try:
                off_x = np.random.randint(0, max(1, (cols-self.input_size)//2))
                off_y = np.random.randint(0, max(1, (rows-self.input_size)//2))
                random_w = np.random.randint(self.input_size, max(self.input_size+1,
                                                                  cols - off_x))
                random_h = np.random.randint(self.input_size, max(self.input_size+1,
                                                                  rows - off_y))
            except:
                import epdb; epdb.set_trace()
            im = im[off_y: off_y+random_h,
                    off_x: off_x+random_w, :]
            im = cv2.resize(im, (self.input_size, self.input_size))
        # BGR 2 RGB
        im = im[:, :, ::-1].copy()
        # print im.shape
        # im = preprocessing.contrast_enhanced(im)

        # mean = [0.3940, 0.2713, 0.1869], RGB
        # std = [0.2777, 0.1981, 0.1574]
        # mean and std, of imaterialist, are
        # 0.5883, 0.5338, 0.5273 and
        # 0.3363, 0.3329, 0.3268
        im = ToTensor()(im)
        for t, m, s in zip(im,
                           [0.5883, 0.5338, 0.5273],
                           [0.3363, 0.3329, 0.3268]):
            t.sub_(m).div_(s)
        label = np.zeros([228], dtype=np.float32)
        label[self.im_labels[index]] = 1
        return im, label

    def __len__(self):
        return len(self.im_filenames)
