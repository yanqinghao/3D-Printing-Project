# coding=utf-8
from __future__ import absolute_import, print_function, division

import os
import numpy as np
import tensorflow as tf
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Folder, Model, Json
from suanpan.model import Model as BaseModel
from suanpan.storage import storage
from suanpan.utils import image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from utils.prework import get_predict_file, get_od_image


class ODModel(BaseModel):
    def __init__(self):
        super(ODModel, self).__init__()
        self.model_dir = "./output_model"
        self.model_init = False

    def load(self, path):
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_FROZEN_GRAPH = os.path.join(path, "frozen_inference_graph.pb")
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join(path, "nozzle_label_map.pbtxt")
        ## Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
        category_index = label_map_util.create_category_index_from_labelmap(
            PATH_TO_LABELS, use_display_name=True
        )
        self.category_index = category_index
        self.detection_graph = detection_graph
        return path

    def predict(self, X):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        X_expanded = np.expand_dims(X, axis=0)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(
            X_expanded, self.detection_graph
        )
        # Visualization of the results of a detection.
        image_boxes = vis_util.visualize_boxes_and_labels_on_image_array(
            X,
            output_dict["detection_boxes"],
            output_dict["detection_classes"],
            output_dict["detection_scores"],
            self.category_index,
            instance_masks=output_dict.get("detection_masks"),
            use_normalized_coordinates=True,
            line_thickness=8,
        )
        return output_dict, image_boxes

    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    "num_detections",
                    "detection_boxes",
                    "detection_scores",
                    "detection_classes",
                    "detection_masks",
                ]:
                    tensor_name = key + ":0"
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name
                        )
                if "detection_masks" in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict["detection_boxes"], [0])
                    detection_masks = tf.squeeze(tensor_dict["detection_masks"], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict["num_detections"][0], tf.int32
                    )
                    detection_boxes = tf.slice(
                        detection_boxes, [0, 0], [real_num_detection, -1]
                    )
                    detection_masks = tf.slice(
                        detection_masks, [0, 0, 0], [real_num_detection, -1, -1]
                    )
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[1], image.shape[2]
                    )
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8
                    )
                    # Follow the convention by adding back the batch dimension
                    tensor_dict["detection_masks"] = tf.expand_dims(
                        detection_masks_reframed, 0
                    )
                image_tensor = tf.get_default_graph().get_tensor_by_name(
                    "image_tensor:0"
                )

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict["num_detections"] = int(output_dict["num_detections"][0])
                output_dict["detection_classes"] = output_dict["detection_classes"][
                    0
                ].astype(np.int64)
                output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
                output_dict["detection_scores"] = output_dict["detection_scores"][0]
                if "detection_masks" in output_dict:
                    output_dict["detection_masks"] = output_dict["detection_masks"][0]
        return output_dict


@app.input(Folder(key="inputData1", alias="predictImages"))
@app.input(Model(key="inputModel2", alias="model", type=ODModel))
@app.output(Json(key="outputData1", alias="predictions"))
@app.output(Folder(key="outputData2", alias="images"))
def SPObjectDetectionPrediction(context):
    args = context.args
    predict_image = get_predict_file(args.predictImages)
    bboxes = []
    i = 0
    for inputImage in get_od_image(predict_image):
        predictedBox, imageBox = args.model.predict(inputImage)
        bboxes.append(predictedBox)
        image.save(
            os.path.join(
                args.images,
                storage.delimiter.join(predict_image[i].split(storage.delimiter)[8:]),
            ),
            imageBox[:, :, ::-1],
        )
        i += 1
    return bboxes, args.images


@app.afterCall
def modelHotReload(context):
    args = context.args
    if app.isStream:
        args.model.reload(duration=args.duration)


if __name__ == "__main__":
    suanpan.run(app)
