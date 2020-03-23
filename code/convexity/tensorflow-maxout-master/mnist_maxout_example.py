from __future__ import print_function

import sys

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def create_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initializer(shape=shape), name=name)


def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=name)


tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 100
display_step = 1
logs_path = '/tmp/tensorflow_logs/example'
lin_path = lambda alpha, W1_0, W1_f: (1 - alpha) * W1_0 + alpha * W1_f 

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')


def linear():
    W1 = create_weight_variable('Weights', [784, 10])
    b1 = create_bias_variable('Bias', [10])
    return tf.nn.softmax(tf.matmul(x, W1) + b1)


def hidden_relu():
    W1 = create_weight_variable('Weights', [784, 100])
    b1 = create_bias_variable('Bias', [100])

    W2 = create_weight_variable('Weights2', [100, 10])
    b2 = create_bias_variable('Bias2', [10])
    t = tf.nn.relu(tf.matmul(x, W1) + b1)
    return tf.nn.softmax(tf.matmul(t, W2) + b2)


def hidden_maxout():
    W1 = create_weight_variable('Weights', [784, 100])
    b1 = create_bias_variable('Bias', [100])

    W2 = create_weight_variable('Weights2', [50, 10])
    b2 = create_bias_variable('Bias2', [10])

    from maxout import max_out
    t = max_out(tf.matmul(x, W1) + b1, 50)
    return (W1, b1, W2, b2, tf.nn.softmax(tf.matmul(t, W2) + b2))


def usage_type():
    usage = 'Usage: python mnist_maxout_example.py (LINEAR|RELU|MAXOUT) (TRAIN|LOAD)'
    assert len(sys.argv) == 3, usage
    t = sys.argv[1].upper()
    print('Type = ' + t)
    res = {'activation': None, 'usage': None}
    if t == 'LINEAR':
        res['activation'] = linear()
    elif t == 'RELU':
        res['activation'] = hidden_relu()
    elif t == 'MAXOUT':
        res['activation'] = hidden_maxout()
    else:
        raise Exception('Unknown type. ' + usage)

    u = sys.argv[2].upper()
    print('Usage = ' + u)
    if u == 'TRAIN':
        res['usage'] = 'TRAIN'
    elif u == 'LOAD':
        res['usage'] = 'LOAD'
    else:
        raise Exception('Unknown type. ' + usage)
    if res['activation'] is not None and res['usage']:
        return (res['activation'], res['usage'])
# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    params, usage = usage_type()
    W1, b1, W2, b2, pred = params 
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

is_train = usage == 'TRAIN'
is_load = usage == 'LOAD'

saver = tf.train.Saver(max_to_keep=None)
if is_train:
    # Initializing the variables
    init = tf.initialize_all_variables()
    # Create a summary to monitor cost tensor
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar('accuracy', acc)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
elif is_load:
    sess_0 = tf.Session()
    saver.restore(sess_0, 'models/init_model/mnist_maxout-0')
    W1_0, b1_0, W2_0, b2_0 = sess_0.run([W1, b1, W2, b2])
    print('Init vars: ', W1_0, b1_0, W2_0, b2_0)

    sess_f = tf.Session()
    saver.restore(sess_f, 'models/last_model/mnist_maxout-100')
    W1_f, b1_f, W2_f, b2_f = sess_f.run([W1, b1, W2, b2])
    print('Init vars: ', W1_f, b1_f, W2_f, b2_f)
        
# Launch the graph
with tf.Session() as sess:
    if is_train:
        
        sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        saver.save(sess, 'models/init_model/mnist_maxout', global_step = 0)
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):

                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop), cost op (to get loss value)
                # and summary nodes
                _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                         feed_dict={x: batch_xs, y: batch_ys})
                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * total_batch + i)
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
            if (epoch + 1) == 100:
                saver.save(sess, 'models/last_model/mnist_maxout', global_step = epoch+1)

        print('Optimization Finished!')

        # Test model
        # Calculate accuracy
        print('Accuracy:', acc.eval({x: mnist.test.images, y: mnist.test.labels}))

        print('Run the command line:')
        print('--> tensorboard --logdir=/tmp/tensorflow_logs')
        print('Then open http://0.0.0.0:6006/ into your web browser')
    if is_load:
        losses_train = []
        losses_test = []
        for alpha in np.linspace(0, 1, 200):
            losses_train.append(sess.run([cost], feed_dict = {x: mnist.train.images, y: mnist.train.labels, W1: lin_path(alpha, W1_0, W1_f), b1: lin_path(alpha, b1_0, b1_f), W2: lin_path(alpha, W2_0, W2_f), b2: lin_path(alpha, b2_0, b2_f) }))
            losses_test.append(sess.run([cost], feed_dict = {x: mnist.test.images, y: mnist.test.labels, W1: lin_path(alpha, W1_0, W1_f), b1: lin_path(alpha, b1_0, b1_f), W2: lin_path(alpha, W2_0, W2_f), b2: lin_path(alpha, b2_0, b2_f) }))

        plt.plot(np.linspace(0, 1, 200), losses_train, label = 'train')
        plt.plot(np.linspace(0, 1, 200), losses_test, label = 'test')
        plt.xlabel('alphas')
        plt.ylabel('loss')
        plt.legend(loc='upper right',scatterpoints=1,fontsize=8)
        plt.savefig('fig/mnist_maxout.png')
