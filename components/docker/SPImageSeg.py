# coding=utf-8
from __future__ import absolute_import, print_function

import os
import math
from suanpan.app import app
from suanpan.app.arguments import Int
from suanpan.storage import storage
from suanpan.utils import image
from arguments import Images


@app.input(Images(key="inputImage"))
@app.param(Int(key="x", default=100))
@app.param(Int(key="y", default=100))
@app.output(Images(key="outputImage"))
def SPImageSeg(context):
    args = context.args
    images = args.inputImage

    for idx, img in enumerate(images):
        splitX, splitY = (
            math.ceil(img.shape[1] / args.x),
            math.ceil(img.shape[0] / args.y),
        )
        i = 0
        for m in range(splitX):
            for n in range(splitY):
                filename = (
                    os.path.splitext(
                        storage.delimiter.join(
                            images.images[idx].split(storage.delimiter)[8:]
                        )
                    )[0]
                    + "_"
                    + str(i)
                    + ".png"
                )
                image.save(
                    os.path.join(args.outputImage, filename,),
                    img[
                        n * args.y : (n + 1) * args.y, m * args.x : (m + 1) * args.x, :,
                    ],
                )
                i += 1

    return args.outputImage


if __name__ == "__main__":
    SPImageSeg()
