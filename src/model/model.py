from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
import tensorflow as tf
import random
import numpy as np
import csv
import sklearn.model_selection as sk
import numbers

tf.set_random_seed(486)


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def dropout_selu(x, keep_prob, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""
    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
                                lambda: dropout_selu_impl(x, keep_prob, alpha, noise_shape, seed, name),
                                lambda: array_ops.identity(x))


def MinMaxScaler(data, min, max):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - min
    denominator = max - min
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


def run():
    data = np.loadtxt("All_data_preprocessing_upsampling.csv", delimiter=",", dtype=np.float32)
    data2 = np.loadtxt("test_data_prediction.csv", delimiter=",", dtype=np.float32)
    x_data = data[:,2:8]
    x_data_min = np.min(x_data, 0)
    x_data_max = np.max(x_data, 0)
    x_data = MinMaxScaler(x_data,x_data_min,x_data_max)

    x_data_test = data2[:, 2:8]
    x_data_test = MinMaxScaler(x_data_test, x_data_min, x_data_max)

    y_data = data[:, [-1]]
    x_train, x_test, y_train, y_test = sk.train_test_split(x_data, y_data, test_size=0.2, random_state=486)

    learning_rate = 0.01
    training_steps = 300
    num_hidden_1 = 12
    num_hidden_2 = 48
    num_hidden_3 = 96
    num_hidden_4 = 96
    num_hidden_5 = 48
    num_hidden_6 = 12
    drop_out_prob = 0.5
    beta = 0
    batch_size = 1000
    nb_feature = 6
    nb_classes = 4

    X = tf.placeholder(tf.float32, [None, nb_feature])
    Y = tf.placeholder(tf.int32, [None, 1])

    Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
    Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

    keep_prob = tf.placeholder(tf.float32)

    W1 = tf.get_variable("W1", shape=[nb_feature, num_hidden_1], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([num_hidden_1]))
    L1 = selu(tf.matmul(X, W1) + b1)
    L1 = dropout_selu(L1, keep_prob=keep_prob)

    W2 = tf.get_variable("W2", shape=[num_hidden_1, num_hidden_2],initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([num_hidden_2]))
    L2 = selu(tf.matmul(L1, W2) + b2)
    L2 = dropout_selu(L2, keep_prob=keep_prob)

    W3 = tf.get_variable("W3", shape=[num_hidden_2, num_hidden_3],initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([num_hidden_3]))
    L3 = selu(tf.matmul(L2, W3) + b3)
    L3 = dropout_selu(L3, keep_prob=keep_prob)

    W4 = tf.get_variable("W4", shape=[num_hidden_3, num_hidden_4],initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([num_hidden_4]))
    L4 = selu(tf.matmul(L3, W4) + b4)
    L4 = dropout_selu(L4, keep_prob=keep_prob)

    W5 = tf.get_variable("W5", shape=[num_hidden_4, num_hidden_5],initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([num_hidden_5]))
    L5 = selu(tf.matmul(L4, W5) + b5)
    L5 = dropout_selu(L5, keep_prob=keep_prob)

    W6 = tf.get_variable("W6", shape=[num_hidden_5, num_hidden_6],initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.Variable(tf.random_normal([num_hidden_6]))
    L6 = selu(tf.matmul(L5, W6) + b6)
    L6 = dropout_selu(L6, keep_prob=keep_prob)

    W7 = tf.get_variable("W7", shape=[num_hidden_6, nb_classes],initializer=tf.contrib.layers.xavier_initializer())
    b7 = tf.Variable(tf.random_normal([nb_classes]))
    logits = tf.matmul(L6, W7) + b7
    hypothesis = tf.nn.softmax(logits)
    regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) + tf.nn.l2_loss(W6) + tf.nn.l2_loss(W7)

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tf.stop_gradient([Y_one_hot]))
    cost = tf.reduce_mean(cost_i) + beta * regularizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    prediction = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    C_matrix = tf.confusion_matrix(predictions=tf.argmax(hypothesis, 1), labels=tf.argmax(Y_one_hot, 1))

    total_batch = int(len(x_train) / batch_size)
    print(total_batch)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for epoch in range(training_steps):
        avg_cost = 0
        for i in range(total_batch):
            feed_dict = {X: x_train[i*batch_size:(i+1)*batch_size,:], Y: y_train[i*batch_size:(i+1)*batch_size,:], keep_prob: drop_out_prob}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
        acc_train = sess.run(accuracy, feed_dict={X: x_train, Y: y_train, keep_prob: 1})
        acc_test = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob: 1})
        print('Epoch:{:4d}\tCost = {:.9f}\t Acc_train = {:.2%}\t Acc_test = {:.2%}'.format(epoch+1, avg_cost, acc_train, acc_test))

    print('Learning Finished!')

    prediction = sess.run([hypothesis], feed_dict = {X: x_data_test, keep_prob: 1})
    print(prediction)
    confusion = sess.run(C_matrix, feed_dict={X: x_data, Y: y_data, keep_prob: 1})
    print(confusion)

    coord.request_stop()
    coord.join(threads)

    f = open('output.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for a in range(len(prediction[0])):
         wr.writerow(prediction[0][a])
    f.close()
    f = open('confusion.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for a in range(len(confusion)):
         wr.writerow(confusion[a])
    f.close()
