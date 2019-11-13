# coding=utf-8
from __future__ import absolute_import, print_function

import os
from PIL import Image

from suanpan.components import Result
from suanpan.storage.arguments import Folder
from suanpan.utils import image
import numpy as np


class ImageLoader(object):
    IMG_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )

    def __init__(self, path):
        self.folder = path
        self.images = self.find_all_images(self.folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        imageArr = self.loader(self.images[item])
        return imageArr

    def loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return np.asarray(img.convert("RGB"))

    def find_all_images(self, folder):
        files_ = []
        list = [i for i in os.listdir(folder)]
        for i in range(0, len(list)):
            path = os.path.join(folder, list[i])
            if os.path.isdir(path):
                files_.extend(self.find_all_images(path))
            if not os.path.isdir(path):
                if path.lower().endswith(self.IMG_EXTENSIONS):
                    files_.append(path)
        return files_

    def numpy_saver(self, data, name):
        if len(data.shape) == 4:
            image.saves(os.path.join(self.folder, name), data)
        elif len(data.shape) == 3:
            image.save(os.path.join(self.folder, name + ".png"), data)
        else:
            raise "Wrong type, make sure your image shape=(n,h,w,c) or (h,w,c)"

    def list_saver(self, data, name):
        if isinstance(data[0], np.ndarray) and len(data[0].shape) == 3:
            image.saves(os.path.join(self.folder, name), data)
        elif isinstance(data[0], tuple) and len(data[0]) == 2:
            for path, img in data:
                image.save(os.path.join(self.folder, path), img)
        else:
            raise "Wrong type, make sure your image shape=(n,h,w,c)"


class Images(Folder):
    def format(self, context):
        super(Images, self).format(context)
        if self.folderPath:
            self.value = ImageLoader(self.folderPath)
        else:
            self.value = None

        return self.value

    def save(self, context, result):

        if isinstance(result.value, (str, Folder)):
            pass
        elif isinstance(result.value, np.ndarray):
            value = ImageLoader(self.folderPath)
            value.numpy_saver(result.value, "images")
        elif isinstance(result.value, list):
            value = ImageLoader(self.folderPath)
            value.list_saver(result.value, "images")
        else:
            raise "Images type need str or np.ndarray"

        return super(Images, self).save(context, Result.froms(value=self.folderPath))

