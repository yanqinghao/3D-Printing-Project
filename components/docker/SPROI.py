# coding=utf-8
from __future__ import absolute_import, print_function

import os
import numpy as np
from PIL import Image
from suanpan.app import app
from suanpan.app.arguments import Folder, Json
from suanpan.utils import image as im


@app.input(Folder(key="inputData1"))
@app.input(Json(key="inputData2"))
@app.output(Folder(key="outputData1"))
def SPROI(context):
    args = context.args
    image_folder = args.inputData1
    detection_dict = args.inputData2
    images = os.listdir(image_folder)
    for i, img in enumerate(images):
        image = Image.open(os.path.join(image_folder, img))
        im_height, im_width = image.size
        boxes = detection_dict[i]["detection_boxes"]
        y_min = boxes[0][0] * im_height
        x_min = boxes[0][1] * im_width
        y_max = boxes[0][2] * im_height
        x_max = boxes[0][3] * im_width

        x_2 = int((x_min + x_max) / 2) + 72
        y_1 = int(y_max + 10)
        x_1 = int(x_2 - 72)
        y_2 = int(y_1 + 72)
        box = (x_1, y_1, x_2, y_2)
        roi = image.crop(box)
        im.save(os.path.join(args.outputData1, img), np.array(roi)[:, :, -1])

    return args.outputData1


if __name__ == "__main__":
    SPROI()
