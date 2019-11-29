# coding=utf-8
from __future__ import absolute_import, print_function, division

import os
from absl import flags
from google.protobuf import text_format
import tensorflow as tf
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Folder, Model, Int
from suanpan.model import Model as BaseModel
from suanpan import path as P
from object_detection import exporter, model_hparams, model_lib
from object_detection.protos import pipeline_pb2


class ODModel(BaseModel):
    def __init__(self):
        super(ODModel, self).__init__()
        self.model_dir = "./output_model"

    def load(self, path):
        return path

    def save(self, path):
        P.copy(self.model_dir, path)
        return path

    def train(self, X, y, epochs=10):
        pass

    def evaluate(self, X, y):
        pass

    def predict(self, X):
        pass


@app.input(Folder(key="inputData1", alias="data"))
@app.input(Folder(key="inputData2", alias="model_config"))
@app.param(Int(key="param1", alias="num_train_steps", default=1))
@app.output(Model(key="outputModel1", alias="model", type=ODModel))
def SPObjectDetection(context):
    args = context.args
    P.copy(args.data, "./data")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 指定第一块GPU可用
    sess = tf.Session()
    FLAGS = {}
    FLAGS["model_dir"] = "./model"
    FLAGS["pipeline_config_path"] = args.model_config + "/ssd_mobilenet_v1_coco.config"
    FLAGS["num_train_steps"] = args.num_train_steps
    FLAGS["eval_training_data"] = False
    FLAGS["sample_1_of_n_eval_examples"] = 1
    FLAGS["sample_1_of_n_eval_on_train_examples"] = 5
    FLAGS["hparams_overrides"] = None
    FLAGS["checkpoint_dir"] = None
    FLAGS["run_once"] = False
    FLAGS["input_type"] = "image_tensor"
    FLAGS["input_shape"] = None
    FLAGS["trained_checkpoint_prefix"] = "./model"
    FLAGS["output_directory"] = "./output_model"
    FLAGS["config_override"] = ""
    FLAGS["write_inference_graph"] = False

    config = tf.estimator.RunConfig(model_dir=FLAGS["model_dir"])

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS["hparams_overrides"]),
        pipeline_config_path=FLAGS["pipeline_config_path"],
        train_steps=FLAGS["num_train_steps"],
        sample_1_of_n_eval_examples=FLAGS["sample_1_of_n_eval_examples"],
        sample_1_of_n_eval_on_train_examples=(
            FLAGS["sample_1_of_n_eval_on_train_examples"]
        ),
    )
    estimator = train_and_eval_dict["estimator"]
    train_input_fn = train_and_eval_dict["train_input_fn"]
    eval_input_fns = train_and_eval_dict["eval_input_fns"]
    eval_on_train_input_fn = train_and_eval_dict["eval_on_train_input_fn"]
    predict_input_fn = train_and_eval_dict["predict_input_fn"]
    train_steps = train_and_eval_dict["train_steps"]

    if FLAGS["checkpoint_dir"]:
        if FLAGS["eval_training_data"]:
            name = "training_data"
            input_fn = eval_on_train_input_fn
        else:
            name = "validation_data"
            # The first eval input will be evaluated.
            input_fn = eval_input_fns[0]
        if FLAGS["run_once"]:
            estimator.evaluate(
                input_fn,
                steps=None,
                checkpoint_path=tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]),
            )
        else:
            model_lib.continuous_eval(
                estimator, FLAGS["checkpoint_dir"], input_fn, train_steps, name
            )
    else:
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False,
        )

        # Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(FLAGS["pipeline_config_path"], "r") as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(FLAGS["config_override"], pipeline_config)
    if FLAGS["input_shape"]:
        input_shape = [
            int(dim) if dim != "-1" else None for dim in FLAGS["input_shape"].split(",")
        ]
    else:
        input_shape = None
    exporter.export_inference_graph(
        FLAGS["input_type"],
        pipeline_config,
        FLAGS["trained_checkpoint_prefix"]
        + "/model.ckpt-"
        + str(FLAGS["num_train_steps"]),
        FLAGS["output_directory"],
        input_shape=input_shape,
        write_inference_graph=FLAGS["write_inference_graph"],
    )
    return args.model


if __name__ == "__main__":
    suanpan.run(app)
