import os
import pickle
import numpy as np
import tensorflow as tf
from suanpan.model import Model as BaseModel
from suanpan.storage import storage
from suanpan.log import logger
from suanpan import path as P
from utils.prework import get_file, get_batch, get_image
from utils.model import inference, losses, trainning, evaluation


class TFModel(BaseModel):
    def __init__(self):
        super(TFModel, self).__init__()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 指定第一块GPU可用
        self.model_init = False
        self.label_map = {
            "dot_bulge": 0,
            "dot_hiatus": 1,
            "level_bulge": 2,
            "level_dislocation": 3,
            "level_hiatus": 4,
            "normal": 5,
            "transformation": 6,
        }

    def load(self, path):
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            obj = pickle.load(f)
        attrlist = [
            "n_classes",
            "img_w",
            "img_h",
        ]
        for i in attrlist:
            setattr(self, i, obj[i])
        self.model_init = False
        self.model_dir = path
        return path

    def save(self, path):
        P.copy(self.model_dir, path)
        return path

    def prepare(self, train_dir, **kwargs):
        img_w = kwargs.pop("img_w", 72)
        img_h = kwargs.pop("img_h", 72)
        batch_size = kwargs.pop("batch_size", 64)
        cap = kwargs.pop("cap", 64)

        self.img_w = img_w
        self.img_h = img_h
        self.batch_size = batch_size
        self.cap = cap

        train, train_label = get_file(train_dir, self.label_map)

        # 训练数据及标签
        image_batch, label_batch = get_batch(
            train, train_label, img_w, img_h, batch_size, cap,
        )
        return image_batch, label_batch

    def train(self, image, label, logs_dir, model_dir, **kwargs):
        learning_rate = kwargs.pop("learning_rate", 0.0001)
        n_classes = kwargs.pop("n_classes", 7)
        max_step = kwargs.pop("max_step", 4501)
        decay_steps = kwargs.pop("decay_steps", 233)
        end_learning_rate = kwargs.pop("end_learning_rate", 0.000001)
        power = kwargs.pop("power", 0.5)
        cycle = kwargs.pop("cycle", False)

        self.n_classes = n_classes

        # 训练操作定义
        train_logits = inference(image, n_classes)
        train_loss = losses(train_logits, label)
        train_op = trainning(train_loss, learning_rate)
        train_acc = evaluation(train_logits, label)

        # 这个是log汇总记录
        summary_op = tf.summary.merge_all()

        # 产生一个会话
        sess = tf.Session()
        train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
        # 产生一个saver来存储训练好的模型
        saver = tf.train.Saver()
        # 所有节点初始化
        sess.run(tf.global_variables_initializer())
        # 队列监控
        coord = tf.train.Coordinator()  # 设置多线程协调器
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        P.mkdirs(model_dir)
        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            pickle.dump(
                {
                    "n_classes": self.n_classes,
                    "img_h": self.img_h,
                    "img_w": self.img_w,
                },
                f,
            )
        # 进行batch的训练
        try:
            # 执行MAX_STEP步的训练，一步一个batch
            for step in np.arange(max_step):
                # print(step)
                if coord.should_stop():
                    break

                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
                learning_rate = tf.train.polynomial_decay(
                    learning_rate=learning_rate,
                    global_step=step,
                    decay_steps=decay_steps,
                    end_learning_rate=end_learning_rate,
                    power=power,
                    cycle=cycle,
                )
                cycle = False
                if step % decay_steps == 0:
                    cycle = True
                # 每隔100步打印一次当前的loss以及acc，同时记录log，写入writer
                if step % 100 == 0:
                    # print(step)
                    # print('Step %d, train loss = %.2f, train accuracy = %.2f%%, learnning rate = %f' % (step, tra_loss, tra_acc * 100.0,learning_rate))
                    # learning_rate = learning_rate*0.5
                    train_txt = open(os.path.join(logs_dir, "D3W3S7_train.txt"), "a")
                    train_txt.write(
                        "Step %d, train loss = %.2f, train accuracy = %.2f%% \n"
                        % (step, tra_loss, tra_acc * 100.0)
                    )
                    train_txt.close()

                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                    checkpoint_path = os.path.join(model_dir, "D3W3S7.ckpt")
                    saver.save(sess, checkpoint_path)
            logger.info("Step %d training done" % (step))

        except tf.errors.OutOfRangeError:
            logger.info("Done training -- epoch limit reached")

        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
        self.model_dir = model_dir
        return self.model_dir

    def evaluate(self, eval_dir, logs_dir):
        eval_image, eval_label = get_file(eval_dir, self.label_map)
        predictions = []
        i = 0
        with tf.Graph().as_default():
            for image in get_image(eval_image):
                predictions.append(self.predict(image, self.model_dir))
                if i % 100 == 0:
                    logger.info("eval process {}".format(i))
                i += 1
        eval_txt = open(os.path.join(logs_dir, "D3W3S7_test.txt"), "a")
        for label, num in self.label_map.items():
            eval_txt.write(
                "\n num_{}: {}, predict_{}: {}".format(
                    label,
                    sum([1 for i in eval_label if i == num]),
                    label,
                    sum([1 for i in predictions if i == num]),
                )
            )
        eval_txt.write(
            "\n num_total: {}, num_correct: {}, accuracy= {:.2%} ".format(
                len(eval_label),
                sum([1 for i, j in zip(predictions, eval_label) if i == j]),
                sum([1 for i, j in zip(predictions, eval_label) if i == j])
                / len(eval_label),
            )
        )
        eval_txt.close()

    def predict(self, X, model_dir):
        if not self.model_init:
            p = inference(X, self.n_classes)
            self.logits = tf.nn.softmax(p)
            saver = tf.train.Saver()
            self.sess = tf.Session()
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.model_init = True
        x = tf.placeholder(tf.float32, shape=[1, self.img_h, self.img_w, 3])
        prediction = self.sess.run(
            self.logits, feed_dict={x: X.eval(session=self.sess)}
        )
        return np.argmax(prediction, axis=1)[0]
