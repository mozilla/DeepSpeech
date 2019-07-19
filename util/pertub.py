import tensorflow as tf
import pyrubberband

is_speed = True
    
def speed_pertub(samples, fs):
    samples = samples.flatten().astype('float32')
    speed_range = np.random.uniform(low=0.8, high=1.4)
    return pyrubberband.pyrb.time_stretch(samples, fs, speed_range).astype('float32')

def pertub(decoded):
    samples = decoded.audio
    fs = decoded.sample_rate
    if is_speed:
        samples = tf.py_func(speed_pertub,[samples, fs], tf.float32)
    return samples