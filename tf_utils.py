import tensorflow as tf
import numpy as np
import data_utils
import os

INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

def load_inception(model_path):
    with open(model_path, 'rb') as model:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model.read())
        input_tensor, inception_tensor = tf.import_graph_def(
            graph_def,
            name = '',
            return_elements = [
                'DecodeJpeg/contents:0',
                'pool_3:0'
            ])

    return input_tensor, inception_tensor

def get_features(sess, images, input_image_op, inception_tensor, label_to_index, model_dir):
    batch_size = len(images)
    num_labels = len(label_to_index)
    features_size = inception_tensor.get_shape()[0]

    features = np.zeros((batch_size, features_size))
    labels = np.zeros((batch_size, num_labels))

    for i, image in enumerate(images):
        if (model_dir):
            feature_cache_dir = os.path.join(model_dir, 'cache')
        else:
            feature_cache_dir = None

        if not os.path.exists(feature_cache_dir):
            os.makedirs(feature_cache_dir)

        feature_cache_path = os.path.join(feature_cache_dir, image['image_path'].split(os.sep)[-1] + '.feat')
        feature_cache = data_utils.load_array(feature_cache_path)

        if feature_cache:
            features[i, :] = feature_cache
        else:
            image_data = data_utils.read_image(image['image_path'])
            feature = sess.run(
                inception_tensor,
                feed_dict = {input_image_op: image_data})
            data_utils.save_array(feature, feature_cache_path)
            features[i, :] = feature

        label = image['label']
        labels[i, label_to_index[label]] = 1

    return (features, labels)
