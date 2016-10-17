import tensorflow as tf
import numpy as np
import data_utils
import argparse
import os

def main(args):
    image_data = data_utils.read_image(args.image_path)
    saver = tf.train.import_meta_graph(args.model_path + '.meta')
    index_to_label = data_utils.load_index_file(os.path.join(args.label_path, 'index_to_label'))

    with tf.Session() as sess:
        saver.restore(sess, args.model_path)

        input_image_op = tf.get_default_graph().get_tensor_by_name('DecodeJpeg/contents:0')
        inception_tensor = tf.get_default_graph().get_tensor_by_name('pool_3:0')
        inception_tensor = tf.reshape(inception_tensor, [-1])

        input_tensor = tf.get_default_graph().get_tensor_by_name('input_tensor:0')
        result_index = tf.get_default_graph().get_tensor_by_name('result_index:0')

        features = sess.run(
            inception_tensor,
            feed_dict = {input_image_op: image_data})

        features = np.reshape(features, (1, len(features)))   

        result = sess.run(
            result_index,
            feed_dict = {input_tensor: features})
    
        print('Result: %s' % index_to_label[result[0]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model')
    parser.add_argument(
        '-l',
        '--labels-path',
        help='Path to where the index_to_label file is. Defaults to "training_data/Images"',
        default='training_data/Images',
        dest='label_path')
    parser.add_argument(
        '-m',
        '--model-path',
        help='Path where the model is. Defaults to model/model.graph',
        default='model/model.graph',
        dest='model_path')
    parser.add_argument(
        '-i',
        '--image-path',
        help='Path to the image to classify.',
        dest='image_path',
        required=True)
    main(parser.parse_args())
