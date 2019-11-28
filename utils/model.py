"""
Model.py
参考VGG

"""
import tensorflow as tf


def variable_with_weight_loss(shape, stddev, w1):
    var = tf.truncated_normal(shape, stddev=stddev)
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)
    return var


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.shape[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            scope + "w",
            shape=[kh, kw, n_in, n_out],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        )
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding="SAME")
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, name="b")
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            scope + "w",
            shape=[n_in, n_out],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
        )
        biases = tf.Variable(
            tf.constant(0.1, shape=[n_out], dtype=tf.float32), name="b"
        )
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(
        input_op,
        ksize=[1, kh, kw, 1],
        strides=[1, dh, dw, 1],
        padding="SAME",
        name=name,
    )


def weight_variable(shape, n):
    initial = tf.truncated_normal(shape, stddev=n, dtype=tf.float32)
    return initial


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return initial


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x, name):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name
    )


def inference(images, n_classes):

    p = []
    conv1_1 = conv_op(images, name="conv1_1", kh=7, kw=7, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=7, kw=7, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

    conv2_1 = conv_op(pool1, name="conv2_1", kh=5, kw=5, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=5, kw=5, n_out=128, dh=1, dw=1, p=p)
    conv2_3 = conv_op(conv2_2, name="conv2_3", kh=5, kw=5, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_3, name="pool2", kh=2, kw=2, dw=2, dh=2)

    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_4 = conv_op(conv3_3, name="conv3_4", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_4, name="pool3", kh=2, kw=2, dw=2, dh=2)

    shape = pool3.get_shape()
    flattened_shape = shape[1].value * shape[2].value * shape[3].value
    reshape_1 = tf.reshape(pool3, [-1, flattened_shape], name="reshape_1")

    fc4 = fc_op(reshape_1, name="fc4", n_out=4096, p=p)
    fc4_drop = tf.nn.dropout(fc4, 0.5, name="fc4_drop")  # keep_prob= 0.5

    fc5 = fc_op(fc4_drop, name="fc5", n_out=1024, p=p)
    fc5_drop = tf.nn.dropout(fc5, 0.5, name="fc5_drop")

    fc6 = fc_op(fc5_drop, name="fc6", n_out=n_classes, p=p)
    # softmax= tf.nn.softmax(fc6)
    logits = fc6

    return logits


def losses(logits, labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name="xentropy_per_example"
        )
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "/loss", loss)
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
        tf.summary.scalar(scope.name + "/accuracy", accuracy)
    return accuracy

