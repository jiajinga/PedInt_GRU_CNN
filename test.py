# 这个是针对有预训练模型的情况
from action_predict import action_prediction
from pie_data import PIE
from jaad_data import JAAD
import os
import sys
import yaml
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    )

def test_model(saved_files_path=None):

    if not saved_files_path:
        raise ValueError("saved_files_path is required (folder containing configs.yaml/model files)")
    if not os.path.isdir(saved_files_path):
        raise FileNotFoundError(f"saved_files_path does not exist or is not a directory: {saved_files_path}")
    configs_path = os.path.join(saved_files_path, 'configs.yaml')
    if not os.path.isfile(configs_path):
        raise FileNotFoundError(f"Missing configs.yaml under saved_files_path: {configs_path}")

    with open(configs_path, 'r') as yamlfile:
        opts = yaml.safe_load(yamlfile)
    print(opts)
    model_opts = opts['model_opts']
    if 'data_opts' in opts and opts['data_opts'] is not None:
        data_opts = opts['data_opts'].copy()
    else:
        dataset_name = model_opts.get('dataset', 'pie')
        if dataset_name == 'pie':
            default_cfg_path = 'config_files_pie/configs_default.yaml'
        elif dataset_name == 'jaad':
            default_cfg_path = 'config_files/configs_default.yaml'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        with open(default_cfg_path, 'r') as f:
            default_opts = yaml.safe_load(f)

        if 'data_opts' not in default_opts:
            raise KeyError(f"data_opts not found in default config: {default_cfg_path}")

        data_opts = default_opts['data_opts'].copy()

    if 'net_opts' in opts and opts['net_opts'] is not None:
        net_opts = opts['net_opts'].copy()
    else:
        dataset_name = model_opts.get('dataset', 'pie')
        if dataset_name == 'pie':
            default_net_path = 'config_files_pie/GC_PIE_small.yaml'
        elif dataset_name == 'jaad':
            default_net_path = 'config_files/GC_jaad_small.yaml'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        with open(default_net_path, 'r') as f:
            default_net_opts = yaml.safe_load(f)

        if 'net_opts' not in default_net_opts:
            raise KeyError(f"net_opts not found in default config: {default_net_path}")

        net_opts = default_net_opts['net_opts'].copy()

    tte = model_opts['time_to_event'] if isinstance(model_opts['time_to_event'], int) else \
                model_opts['time_to_event'][1]
    data_opts['min_track_size'] = model_opts['obs_length'] + tte

    if model_opts['dataset'] == 'pie':
        pie_path = os.environ.get('PIE_PATH', 'PIE')
        imdb = PIE(data_path=pie_path)
    elif model_opts['dataset'] == 'jaad':
        jaad_path = os.environ.get('JAAD_PATH', 'JAAD')
        imdb = JAAD(data_path=jaad_path)
    else:
        raise ValueError("{} dataset is incorrect".format(model_opts['dataset']))

    method_class = action_prediction(model_opts['model'])(**net_opts)
    #beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
    #saved_files_path = method_class.train(beh_seq_train, **train_opts, model_opts=model_opts)

    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
    acc, auc, f1, precision, recall = method_class.test(beh_seq_test, saved_files_path)
    print('test done')
    print(acc, auc, f1, precision, recall)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python test.py <saved_files_path>')
        sys.exit(2)
    saved_files_path = sys.argv[1]
    test_model(saved_files_path=saved_files_path)