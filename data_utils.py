import os
import sys
import urllib.request
import tarfile
import random
import json

def download_and_extract_tar(url, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    filename = url.split(os.sep)[-1]
    filepath = os.path.join(dir, filename)

    if not os.path.exists(filepath):
        def download_progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%% [%.2f/%.2f] MB' %
                (
                    filename,
                    float(count * block_size) / float(total_size) * 100.0,
                    count * block_size / 1000000,
                    total_size / 1000000
                )
            )
            sys.stdout.flush()

        urllib.request.urlretrieve(url, filepath, download_progress)
        print('\nDone downloading %s' % filename)
        
        print('Extracting %s...' % filename)

        tar_string = ''
        if filename.endswith('tgz') or filename.endswith('tar.gz'):
            tar_string = 'r:gz'
        elif filename.endswith('tar'):
            tar_string = 'r:'
        tarfile.open(filepath, tar_string).extractall(dir)

        print('Done extracting %s' % filename)
    else:
        print('%s already exists' % filename)

def build_dataset_object(data_dir, **args):
    dataset_cache_file = os.path.join(data_dir, 'dataset.json')
    force_rebuild = args.get('force_rebuild', False)

    if os.path.isfile(dataset_cache_file) and not force_rebuild:
        print("Loading data split and labels from cache")

        with open(dataset_cache_file) as dataset_file:
            dataset = json.load(dataset_file)
            return dataset
    else:
        print("Generating data split and labels")

        test_percent = args.get('test_percent', 0.1)
        validation_percent = args.get('validation_percent', 0.1)

        dataset = {}
        dataset['index_to_label'] = []
        images_and_labels = []

        for folder, _, filenames in os.walk(data_dir):
            if data_dir == folder:
                continue

            image_class = folder.split(os.sep)[-1]
            dataset['index_to_label'].append(image_class)

            for filename in filenames:
                full_path = os.path.join(folder, filename)
                image_and_label = {}
                image_and_label['image_path'] = full_path
                image_and_label['label'] = image_class
                images_and_labels.append(image_and_label)

        dataset['label_to_index'] = dict(
            (v, k) for k, v in enumerate(dataset['index_to_label']))
        random.shuffle(images_and_labels)

        test_slice = int(len(images_and_labels) * test_percent)
        validation_slice = -int(len(images_and_labels) * validation_percent)

        dataset['test'] = images_and_labels[:test_slice]
        dataset['train'] = images_and_labels[test_slice:validation_slice]
        dataset['validation'] = images_and_labels[validation_slice:]

        with open(dataset_cache_file, 'w') as dataset_file:
            json.dump(dataset, dataset_file)

        return dataset

def read_image(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    return image_data

def get_minibatch(data, batch_size):
    return random.sample(data, batch_size)

def save_values(values, filename):
    out_string = ','.join(str(x) for x in values)
    with open(filename, 'w') as out_file:
        out_file.write(out_string)

def load_values(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as in_file:
            in_string = in_file.read()
        values = [float(x) for x in in_string.split(',')]
        return values
    else:
        return None
