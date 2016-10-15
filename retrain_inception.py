import tensorflow as tf
import numpy as np
import data_utils
import argparse
import os

DATA_URL = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

def load_model(model_path):
    with open(model_path, 'rb') as model:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model.read())
        input_tensor, bottleneck_tensor = tf.import_graph_def(
            graph_def,
            name = '',
            return_elements = [
                'DecodeJpeg/contents:0',
                'pool_3:0'
            ])

        return input_tensor, bottleneck_tensor

def get_features(sess, images, input_image_op, bottleneck_tensor, label_to_index, model_dir):
    batch_size = len(images)
    num_labels = len(label_to_index)
    features_size = bottleneck_tensor.get_shape()[0]

    features = np.zeros((batch_size, features_size))
    labels = np.zeros((batch_size, num_labels))

    for i, image in enumerate(images):
        feature_cache_dir = os.path.join(model_dir, 'cache')
        if not os.path.exists(feature_cache_dir):
            os.makedirs(feature_cache_dir)
        feature_cache_path = os.path.join(feature_cache_dir, image['image_path'].split(os.sep)[-1] + '.feat')
        feature_cache = data_utils.load_values(feature_cache_path)

        if feature_cache:
            features[i, :] = feature_cache
        else:
            image_data = data_utils.read_image(image['image_path'])
            feature = sess.run(
                bottleneck_tensor,
                feed_dict = {input_image_op: image_data})
            data_utils.save_values(feature, feature_cache_path)
            features[i, :] = feature

        label = image['label']
        labels[i, label_to_index[label]] = 1

    return (features, labels)

def add_new_layer(features_size, num_labels):
    input_tensor = tf.placeholder(tf.float32, shape=[None, features_size])
    label_tensor = tf.placeholder(tf.float32, shape=[None, num_labels])

    W = tf.get_variable(
        'weights',
        initializer = tf.random_normal_initializer(),
        shape = [features_size, num_labels])

    b = tf.get_variable(
        'biases',
        initializer = tf.constant_initializer(0.0),
        shape = [num_labels])

    logits = tf.matmul(input_tensor, W) + b
    softmax_output = tf.nn.softmax(logits)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits, label_tensor)
    mean_loss = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer().minimize(mean_loss)

    correct_prediction = tf.equal(
        tf.argmax(softmax_output, 1),
        tf.argmax(label_tensor, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return input_tensor, label_tensor, train_step, mean_loss, accuracy

def main(args):
    data_utils.download_and_extract_tar(DATA_URL, args.data_dir)
    data_utils.download_and_extract_tar(INCEPTION_URL, args.model_dir)
    tmp_dir = os.path.join(args.model_dir, 'tmp')

    dataset = data_utils.build_dataset_object(
        os.path.join(args.data_dir, 'Images'),
        test_percent = 0.1,
        training_percent = 0.1,
        force_rebuild = False)

    model_path = (os.path.join(args.model_dir, 'classify_image_graph_def.pb'))
    input_image_op, bottleneck_tensor = load_model(model_path)
    bottleneck_tensor = tf.reshape(bottleneck_tensor, [-1])

    features_size = bottleneck_tensor.get_shape()[0]
    num_labels = len(dataset['label_to_index'])

    input_tensor, label_tensor, train_step, mean_loss, accuracy = \
        add_new_layer(features_size, num_labels)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        checkpoint_state = tf.train.get_checkpoint_state(args.model_dir)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            print('Loading checkpoint')
            saver.restore(sess, checkpoint_state.model_checkpoint_path)

        print('Setting up validation')
        images_validation, labels_validation = \
            get_features(
                sess,
                dataset['validation'],
                input_image_op,
                bottleneck_tensor,
                dataset['label_to_index'],
                args.model_dir)

        for i in range(args.steps):
            images = data_utils.get_minibatch(dataset['train'], args.batch_size)
            features, labels = \
                get_features(
                    sess,
                    images,
                    input_image_op,
                    bottleneck_tensor,
                    dataset['label_to_index'],
                    args.model_dir)
            loss, _ = sess.run(
                [mean_loss, train_step],
                feed_dict = {input_tensor: features,
                             label_tensor: labels})

            print('Step: %i - Loss: %f' % (i + 1, loss))

            if (i + 1) % args.checkpoint_interval == 0:
                print('Saving checkpoint')
                checkpoint_path = os.path.join(args.model_dir, 'model.checkpoint')
                saver.save(
                    sess,
                    checkpoint_path,
                    global_step= i + 1)

                print('Running validation')
                validation_accuracy = sess.run(
                    accuracy,
                    feed_dict = {input_tensor: images_validation,
                                label_tensor: labels_validation})
                print('Validation accuracy: %f%%' % (float(validation_accuracy) * 100.0))

        print('Saving model')
        tf.train.export_meta_graph(filename=os.path.join(args.model_dir, 'model.graph'))

        print('Running test')
        images_test, labels_test = \
            get_features(
                sess,
                dataset['test'],
                input_image_op,
                bottleneck_tensor,
                dataset['label_to_index'],
                args.model_dir)
        test_accuracy = sess.run(
            accuracy,
            feed_dict = {input_tensor: images_test,
                         label_tensor: labels_test})
        print('Test accuracy: %f%%' % (float(test_accuracy) * 100.0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model')
    parser.add_argument(
        '-d',
        '--training-data-dir',
        help='Directory to store the training data. Defaults to "training_data/"',
        default='training_data/',
        dest='data_dir')
    parser.add_argument(
        '-m',
        '--model-dir',
        help='Directory to store the downloaded model and the retrained model',
        default='model/',
        dest='model_dir')
    parser.add_argument(
        '-b',
        '--batch-size',
        help='The batch size',
        default=10,
        type=int,
        dest='batch_size')
    parser.add_argument(
        '-s',
        '--steps',
        help='How many steps to run for',
        default=5000,
        type=int,
        dest='steps')
    parser.add_argument(
        '-c',
        '--checkpoint-interval',
        help='Checkpoint training after these many steps',
        default=100,
        type=int,
        dest='checkpoint_interval')
    main(parser.parse_args())
