import tensorflow as tf
import numpy as np
import data_utils
import argparse
import os

def main(args):
    image_data = data_utils.read_image(args.image_path)
    index_to_label = data_utils.load_index_file(os.path.join(args.label_path, 'index_to_label'))

    saver = tf.train.import_meta_graph(args.model_path + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, args.model_path)

        png_input = tf.get_default_graph().get_tensor_by_name('png_input:0')  
        jpg_input = tf.get_default_graph().get_tensor_by_name('jpg_input:0') 

        decoded_png = tf.get_default_graph().get_tensor_by_name('decoded_png:0')  
        decoded_jpg = tf.get_default_graph().get_tensor_by_name('decoded_jpg:0') 

        decoded_image_input = tf.get_default_graph().get_tensor_by_name('DecodeJpeg:0')
        features_tensor = tf.get_default_graph().get_tensor_by_name('pool_3:0')
        features_tensor = tf.reshape(features_tensor, [-1])

        input_tensor = tf.get_default_graph().get_tensor_by_name('input_tensor:0')
        result_index = tf.get_default_graph().get_tensor_by_name('result_index:0')

        if b'\x89PNG' in image_data[:4]:
            decoded_image = sess.run(
                decoded_png,
                feed_dict = {png_input: image_data})
            features = sess.run(
                features_tensor,
                feed_dict = {decoded_image_input: decoded_image})
        elif b'\xff\xd8\xff' in image_data[:4]:
            decoded_image = sess.run(
                decoded_jpg,
                feed_dict = {jpg_input: image_data})
            features = sess.run(
                features_tensor,
                feed_dict = {decoded_image_input: decoded_image})
        else:
            print('Unsupported image')
            print(image_data[:4])

        features = np.reshape(features, (1, len(features)))   

        result = sess.run(
            result_index,
            feed_dict = {input_tensor: features})

        print('Result: %s' % index_to_label[result[0]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifies an image')
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
