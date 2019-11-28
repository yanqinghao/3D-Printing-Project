# coding=utf-8
from __future__ import absolute_import, print_function

import os
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Folder, Int, Model, Float
from model import TFModel


@app.input(Folder(key="inputData1", alias="trainImages"))
@app.output(Model(key="outputModel1", alias="model", type=TFModel))
@app.output(Folder(key="outputData2", alias="log"))
@app.param(Float(key="param1", alias="lr", default=0.0001))
@app.param(Int(key="param2", alias="bs", default=64))
@app.param(Int(key="param3", alias="max_step", default=1, help="4501"))
@app.param(Int(key="param4", alias="decay_steps", default=233))
@app.param(Float(key="param5", alias="end_learning_rate", default=0.000001))
@app.param(Float(key="param6", alias="power", default=0.5))
@app.param(Float(key="param7", alias="cycle", default=False))
@app.param(Float(key="param8", alias="cap", default=64))
@app.param(Int(key="param9", alias="n_classes", default=7))
@app.param(Int(key="param10", alias="img_w", default=72))
@app.param(Int(key="param11", alias="img_h", default=72))
def SPClassificationTrain(context):
    args = context.args
    prepare_params = {
        "img_w": args.img_w,
        "img_h": args.img_h,
        "batch_size": args.bs,
        "cap": args.cap,
    }
    train_params = {
        "max_step": args.max_step,
        "n_classes": args.n_classes,
        "learning_rate": args.lr,
        "decay_steps": args.decay_steps,
        "end_learning_rate": args.end_learning_rate,
        "power": args.power,
        "cycle": args.cycle,
    }
    train_batch, train_label_batch = args.model.prepare(
        os.path.join(args.trainImages, "train"), **prepare_params
    )
    args.model.train(
        train_batch,
        train_label_batch,
        args.log,
        "./classification-model",
        **train_params
    )
    args.model.evaluate(os.path.join(args.trainImages, "test"), args.log)
    return args.model, args.log


if __name__ == "__main__":
    suanpan.run(app)
