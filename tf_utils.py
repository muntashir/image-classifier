import tensorflow as tf
import numpy as np
import data_utils
import sys
import os

INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

class PretrainedModel:
    def __init__(self, model_path):
        self.png_input = tf.placeholder(tf.string, shape=[], name='png_input')
        self.decoded_png = tf.image.decode_png(self.png_input, channels=3, name='decoded_png')

        self.jpg_input = tf.placeholder(tf.string, shape=[], name='jpg_input')
        self.decoded_jpg = tf.image.decode_jpeg(self.jpg_input, channels=3, name='decoded_jpg')

        with open(model_path, 'rb') as model:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model.read())
            self.decoded_image_input, self.features_tensor = tf.import_graph_def(
                graph_def,
                name = '',
                return_elements = [
                    'DecodeJpeg:0',
                    'pool_3:0'
                ])
            self.features_tensor = tf.reshape(self.features_tensor, [-1])

    def run(self, sess, images, label_to_index, cache_dir):
        batch_size = len(images)
        num_labels = len(label_to_index)
        features_size = self.features_tensor.get_shape()[0]

        features = np.zeros((batch_size, features_size))
        labels = np.zeros((batch_size, num_labels))

        for i, image in enumerate(images):
            if batch_size > 500:
                sys.stdout.write('\rRunning pretrained model on image %i/%i' % (i + 1, batch_size))
                sys.stdout.flush()

            if (cache_dir):
                feature_cache_dir = os.path.join(cache_dir, 'cache')
            else:
                feature_cache_dir = None

            if not os.path.exists(feature_cache_dir):
                os.makedirs(feature_cache_dir)

            feature_cache_path = \
                os.path.join(feature_cache_dir, image['image_path'].split(os.sep)[-1] + '.feat')
            feature_cache = data_utils.load_array(feature_cache_path)

            if feature_cache:
                features[i, :] = feature_cache
            else:
                image_data = data_utils.read_image(image['image_path'])

                if b'\x89PNG' in image_data[:4]:
                    decoded_image = sess.run(
                        self.decoded_png,
                        feed_dict = {self.png_input: image_data})
                    feature = sess.run(
                        self.features_tensor,
                        feed_dict = {self.decoded_image_input: decoded_image})
                elif b'\xff\xd8\xff' in image_data[:4]:
                    decoded_image = sess.run(
                        self.decoded_jpg,
                        feed_dict = {self.jpg_input: image_data})
                    feature = sess.run(
                        self.features_tensor,
                        feed_dict = {self.decoded_image_input: decoded_image})
                else:
                    if batch_size > 500:
                        print()
                    print(image_data[:4])

                data_utils.save_array(feature, feature_cache_path)
                features[i, :] = feature

            label = image['label']
            labels[i, label_to_index[label]] = 1

        if batch_size > 500:
            print()
        return (features, labels)
