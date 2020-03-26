from __future__ import absolute_import, division, print_function, unicode_literals
import sys
from maxout import max_out
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.models import model_from_json

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


def create_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initializer(shape=shape), name=name)


def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=name)


tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# Parameters
learning_rate = .01
training_epochs = 100
batch_size = 100
display_step = 1
adversarial_alpha = .5
epsilon = .25
logs_path = '/tmp/tensorflow_logs/example'


def lin_path(alpha, W1_0, W1_f):
    return (1 - alpha) * W1_0 + alpha * W1_f

# tf Graph Input
# mnist data image of shape 28*28=784


x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')


def linear():
    W1 = create_weight_variable('Weights', [784, 10])
    b1 = create_bias_variable('Bias', [10])

    return tf.nn.softmax(tf.matmul(x, W1) + b1)


def hidden_relu(dropout):
    W1 = create_weight_variable('Weights', [784, 100])
    b1 = create_bias_variable('Bias', [100])

    W2 = create_weight_variable('Weights2', [100, 10])
    b2 = create_bias_variable('Bias2', [10])
    t = tf.nn.relu(tf.matmul(x, W1) + b1)
    if dropout:
        t = tf.nn.dropout(t, rate=0.2)

    return W1, b1, W2, b2, tf.nn.softmax(tf.matmul(t, W2) + b2)


def hidden_maxout(dropout):
    W1 = create_weight_variable('Weights', [784, 100])
    b1 = create_bias_variable('Bias', [100])

    W2 = create_weight_variable('Weights2', [50, 10])
    b2 = create_bias_variable('Bias2', [10])
    t = max_out(tf.matmul(x, W1) + b1, 50)
    if dropout:
        t = tf.nn.dropout(t, rate=0.2)

    return W1, b1, W2, b2, tf.nn.softmax(tf.matmul(t, W2) + b2)


def cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    return model


def usage_type():
    usage_ = 'Usage: python mnist_maxout_example.py (LINEAR|RELU|MAXOUT) (TRAIN|LOAD) (CE|CE_adversarial) (Y|N)'
    assert len(sys.argv) == 5 or (len(sys.argv) == 2 and sys.argv[1].upper() == 'CNN'), usage_

    t = sys.argv[1].upper()
    if t == 'CNN':
        use_cnn_ = True
        return cnn(), '', '', '', '', use_cnn_
    else:
        use_cnn_ = False
        dropout_ = sys.argv[4].upper()
        print('Dropout = ' + dropout_)
        res = {'activation': None, 'usage': None, 'cost': None, 'dropout': None}
        if dropout_ == 'Y':
            res['dropout'] = True
        elif dropout_ == 'N':
            res['dropout'] = False
        else:
            raise Exception('Unknown dropout. ' + dropout_)

        print('Type = ' + t)
        if t == 'LINEAR':
            res['activation'] = linear()
        elif t == 'RELU':
            res['activation'] = hidden_relu(res['dropout'])
        elif t == 'MAXOUT':
            res['activation'] = hidden_maxout(res['dropout'])
        else:
            raise Exception('Unknown type. ' + t)

        u = sys.argv[2].upper()
        print('Usage = ' + u)
        if u == 'TRAIN':
            res['usage'] = 'TRAIN'
        elif u == 'LOAD':
            res['usage'] = 'LOAD'
        else:
            raise Exception('Unknown usage. ' + u)

        declared_cost_ = sys.argv[3].upper()
        print('Cost = ' + declared_cost_)
        if declared_cost_ == 'CE':
            res['cost'] = 'CE'
        elif declared_cost_ == 'CE_ADVERSARIAL':
            res['cost'] = 'CE_ADVERSARIAL'
        else:
            raise Exception('Unknown cost. ' + declared_cost_)

        if res['activation'] is not None and all((res['usage'], res['cost'])):
            return res['activation'], res['usage'], res['cost'], t, res['dropout'], use_cnn_

# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient


params, usage, declared_cost, activation_type, dropout, use_cnn = usage_type()
is_train = usage == 'TRAIN'
is_load = usage == 'LOAD'
init_model_path = 'models/init_model/' + ('cifar_' if use_cnn else 'mnist_') + activation_type + '_' + declared_cost + (
    '_DROPOUT' if dropout else '')
final_model_path = 'models/last_model/' + ('cifar_' if use_cnn else 'mnist_') + activation_type + '_' + declared_cost + (
    '_DROPOUT' if dropout else '')
fig_path = 'fig/' + ('cifar_' if use_cnn else 'mnist_') + activation_type + '_' + declared_cost + ('_DROPOUT' if dropout else '') + '.png'

if use_cnn:
    pass
else:
    with tf.name_scope('Model'):
        # Model
        W1, b1, W2, b2, pred = params
    with tf.name_scope('Loss'):
        # Minimize error using cross entropy
        if declared_cost == 'CE':
            cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
        elif declared_cost == 'CE_ADVERSARIAL':
            J = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
            grad = tf.gradients(J, x)
            fast_gradient = tf.squeeze(tf.sign(grad), [0])
            fast_signed_gradient = epsilon * fast_gradient
            if activation_type == 'MAXOUT':
                pred_adversarial = tf.nn.softmax(tf.matmul(max_out(tf.matmul(x + fast_signed_gradient, W1) + b1, 50), W2) + b2)
            elif activation_type == 'RELU':
                pred_adversarial = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(x, W1) + b1), W2) + b2)
            J_adversarial = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred_adversarial), reduction_indices=1))
            cost = adversarial_alpha * J + (1 - adversarial_alpha) * J_adversarial

    with tf.name_scope('SGD'):
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    with tf.name_scope('Accuracy'):
        # Accuracy
        acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
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
        saver.restore(sess_0, init_model_path + '-0')
        W1_0, b1_0, W2_0, b2_0 = sess_0.run([W1, b1, W2, b2])
        print('Init vars: ', W1_0, b1_0, W2_0, b2_0)

        sess_f = tf.Session()
        saver.restore(sess_f, final_model_path + '-100')
        W1_f, b1_f, W2_f, b2_f = sess_f.run([W1, b1, W2, b2])
        print('Init vars: ', W1_f, b1_f, W2_f, b2_f)


if __name__ == '__main__':
    if use_cnn:
        params.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # serialize model 0 to JSON
        model_json = params.to_json()
        with open(init_model_path + "cnn-0.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        params.save_weights(init_model_path + "cnn-0.h5")
        print("Saved model cnn-0 to disk")

        n_epochs = 10
        history = params.fit(train_images, train_labels, epochs=n_epochs,
                            validation_data=(test_images, test_labels))

        test_loss, test_acc = params.evaluate(test_images, test_labels, verbose=2)
        print('Test accuracy: ', test_acc)

        # serialize last model to JSON
        model_json = params.to_json()
        with open(final_model_path + "cnn-" + str(n_epochs) + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        params.save_weights(final_model_path + "cnn-" + str(n_epochs) + ".h5")
        print("Saved model to disk")

        # load json and create model 0
        json_file = open(init_model_path + "cnn-0.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model_0 = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model_0.load_weights(init_model_path + "cnn-0.h5")
        print("Loaded model from disk")

        # load json and create model 0
        json_file = open(final_model_path + "cnn-" + str(n_epochs) + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model_last = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model_last.load_weights(final_model_path + "cnn-" + str(n_epochs) + ".h5")
        print("Loaded model from disk")

        losses = {'train': [], 'test': []}
        dataset_d = {'train': {'X': train_images, 'Y': train_labels}, 'test': {'X': test_images, 'Y': test_labels}}

        loaded_model_0_weights = [layer.get_weights() for layer in loaded_model_0.layers]
        loaded_model_last_weights = [layer.get_weights() for layer in loaded_model_last.layers]

        def loss_temp(dataset, alpha):
            # evaluate loaded model on test data
            for i in range(len((loaded_model_0_weights))):
                new_weight = [lin_path(alpha, w[0], w[1]) for w in zip(loaded_model_0_weights[i], loaded_model_last_weights[i])]
                loaded_model_last.layers[i].set_weights(new_weight)
            loaded_model_last.compile(optimizer='adam',
                                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                    metrics=['accuracy'])
            loss_, _ = loaded_model_last.evaluate(dataset['X'], dataset['Y'], verbose=2)
            return loss_

        for alpha_ in np.linspace(0, 1, 100):
            print(alpha_)
            for loss_type in ['train', 'test']:
                losses[loss_type].append(loss_temp(dataset_d[loss_type], alpha_))

        plt.plot(np.linspace(0, 1, 100), losses['train'], label='train')
        plt.plot(np.linspace(0, 1, 100), losses['test'], label='test')
        plt.xlabel('alphas')
        plt.ylabel('loss')
        plt.legend(loc='upper right', scatterpoints=1, fontsize=8)
        plt.savefig(fig_path)
        print('Done')

    else:

        # Launch the graph
        with tf.Session() as sess:
            if is_train:

                sess.run(init)

                # op to write logs to Tensorboard
                summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
                saver.save(sess, init_model_path, global_step=0)
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
                        saver.save(sess, final_model_path, global_step = epoch+1)

                print('Optimization Finished!')

                # Test model
                # Calculate accuracy
                print('Accuracy:', acc.eval({x: mnist.test.images, y: mnist.test.labels}))

                print('Run the command line:')
                print('--> tensorboard --logdir=/tmp/tensorflow_logs')
                print('Then open http://0.0.0.0:6006/ into your web browser')
            if is_load:
                losses = {'train': [], 'test': []}
                dataset_d = {'train': mnist.train, 'test': mnist.test}

                def loss_temp(dataset, alpha):
                    return sess.run([cost], feed_dict={x: dataset.images, y: dataset.labels, W1: lin_path(alpha, W1_0, W1_f), b1: lin_path(alpha, b1_0, b1_f), W2: lin_path(alpha, W2_0, W2_f), b2: lin_path(alpha, b2_0, b2_f)})

                for alpha_ in np.linspace(0, 1, 200):
                    for loss_type in ['train', 'test']:
                        losses[loss_type].append(loss_temp(dataset_d[loss_type], alpha_))

                plt.plot(np.linspace(0, 1, 200), losses['train'], label='train')
                plt.plot(np.linspace(0, 1, 200), losses['test'], label='test')
                plt.xlabel('alphas')
                plt.ylabel('loss')
                plt.legend(loc='upper right', scatterpoints=1, fontsize=8)
                plt.savefig(fig_path)
                print('Done')
