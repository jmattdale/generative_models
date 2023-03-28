import yaml

def load_config(config_path):
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    config = sanitize(config)
    return config

def sanitize(config):
    for k, v in config.items():
        if isinstance(v, list):
            if len(v) and isinstance(v[0], list):
                new_v = tuple(tuple(x) for x in v)
            else:
                new_v = tuple(v)
            config[k] = new_v
    return config