import json
import random
from abc import ABCMeta, abstractmethod

class AugmentationPipeline(object):
    def __init__(self, augmentation_config, random_seed=0.001):
        self._rng = random.Random(random_seed)
        self._augmentors, self._rates = self._parse_pipeline_from_json(augmentation_config)
    
    @abstractmethod
    def transform(self, mel_fbank):
        pass

    @abstractmethod
    def _get_augmentor_by_name(self, name, params):
        pass
    
    def _parse_pipeline_from_json(self, config_json):
        try:
            with open(config_json, 'r') as config_file:
                configs = json.load(config_file)
            augmentors = [self._get_augmentor_by_name(config["type"], config["params"]) for config in configs]
            rates = [config["rate"] for config in configs]
        except json.JSONDecodeError as e:
            raise ValueError("Error parsing the audio augmentation pipeline %s" % str(e))
        return augmentors, rates
