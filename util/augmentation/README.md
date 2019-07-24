# Augmentation

## Audio Augmentation

### Augmentation Techniques :
- **Random time stretch**:
 It just shift audio to left/right with a random second. If shifting audio to left (fast forward) with x seconds, first x seconds will mark as 0 (i.e. silence). If shifting audio to right (back forward) with x seconds, last x seconds will mark as 0 (i.e. silence). ``` "type": "speed_pertub"```

### How to Use

```python
from .augmentation import AudioAugmentationPipeline
audio_augmentor = AudioAugmentationPipeline(config_file)
audio_augmentor.transform(samples, fs)
```

where following is the format for confugration file

```json
[ 
    {
        "type": "speed_pertub",
        "params": {"low_speed" : 0.8, "high_speed" : 1.4},
        "rate": 0.3 # fraction of data on which speed_pertubation has to be applied
    }
]
```

## Spectrogram Augmentation

### Augmentation Techniques :
- **Time Warping**:
Time warping is applied via the function sparse image warp of tensorflow. Given a log mel spectrogram with τ time steps, we view it as an image where the time axis is horizontal and the frequency axis is vertical. A random point along the horizontal line passing through the center of the image within the time steps (W, τ − W) is to be warped either to the left or right by a distance w chosen from a uniform distribution from 0 to the time warp parameter W along that line.
``` "type": "time_warp" ```

- **Frequency Mask**:
Frequency masking is applied so that f consecutive mel frequency channels [f0, f0 + f) are masked, where f is first chosen from a uniform distribution from 0 to the frequency mask parameter F, and f0 is chosen from 0, ν − f). ν is the number of mel frequency channels. ``` NOTE: Not Implemented ```

- **Time Mask**:
Time masking is applied so that t consecutive time steps [t0, t0 + t) are masked, where t is first chosen from a
uniform distribution from 0 to the time mask parameter T, and t0 is chosen from [0, τ − t). We introduce an upper bound on the time mask so that a time mask cannot be wider than p times the number of time steps. ``` NOTE: Not Implemented ```

### How to use
```python
from .augmentation import  SpectrumAugmentationPipeline
spectrum_augmentor = SpectrumAugmentationPipeline(config_file)
spectrum_augmentor = spectrum_augmentor.transform(mel_filterbank)
```

where following the the format of configuration file
```json
[ 
    {
        "type": "time_warp",
        "params": {"W" : 80}, #Time Warping Parameter
        "rate": 0.5 # fraction of data on which speed_pertubation has to be applied
    }
]
```

## References
1. https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html
2. https://github.com/shelling203/SpecAugment
3. https://librosa.github.io/librosa/generated/librosa.effects.time_stretch.html
4. https://arxiv.org/pdf/1904.08779.pdf
