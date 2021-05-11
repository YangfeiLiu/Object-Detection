import yaml


def load_config(config_path):
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        dataset_config = config['COMMON']
        train_config = config['TRAIN']
        test_config = config['TEST']
    return {'common': dataset_config, 'train': train_config, 'test': test_config}
