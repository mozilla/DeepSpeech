# Augmentation

## Audio Augmentation
Configuration file

```{json}
[ 
    {
        "type": "speed_pertub",
        "params": {"low_speed" : 0.8, "high_speed" : 1.4},
        "rate": 0.5 # how many times speed_pertubation has to be applied
    }
]
```

Api Usage
```{python}
from .augmentation import AudioAugmentationPipeline
audio_augmentor = AudioAugmentationPipeline(config_file)
audio_augmentor.transform(samples, fs)
```

## Spectrogram Augmentation
Configuration file

```{json}
[ 
    {
        "type": "time_warp",
        "params": {"W" : 80},
        "rate": 0.5 # how many times speed_pertubation has to be applied
    }
]
```

```{python}
from .augmentation import  SpectrumAugmentationPipeline
spectrum_augmentor = SpectrumAugmentationPipeline(config_file)
spectrum_augmentor = mel
```


## References
1. https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html
2. https://github.com/shelling203/SpecAugment
3. https://librosa.github.io/librosa/generated/librosa.effects.time_stretch.html
