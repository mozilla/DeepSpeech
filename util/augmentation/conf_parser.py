import json

def _parse_pipeline_from_json(config_json):
    try:
        with open(config_json, 'r') as config_file:
            configs = json.load(config_file)
        augmentors = [self._get_augmentor_by_name(config["type"], config["params"]) for config in configs]
        rates = [config["rate"] for config in configs]
    except json.JSONDecodeError as e:
        raise ValueError("Error parsing the audio augmentation pipeline %s" % str(e))
    return augmentors, rates