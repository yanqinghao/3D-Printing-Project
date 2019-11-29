# coding=utf-8
from __future__ import absolute_import, print_function

import tensorflow as tf
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Folder, Model, Json
from model import TFModel
from utils.prework import get_predict_file, get_image


@app.input(Folder(key="inputData1", alias="predictImages"))
@app.input(Model(key="inputModel2", alias="model", type=TFModel))
@app.output(Json(key="outputData1", alias="predictions"))
def SPClassificationPredict(context):
    args = context.args
    predict_image = get_predict_file(args.predictImages)
    predictions = []
    with tf.Graph().as_default():
        for image in get_image(predict_image):
            labelnum = args.model.predict(image, args.model.model_dir)
            predictions.append(list(args.model.label_map.keys())[labelnum])

    return predictions


@app.afterCall
def modelHotReload(context):
    args = context.args
    if app.isStream:
        args.model.reload(duration=args.duration)


if __name__ == "__main__":
    suanpan.run(app)
