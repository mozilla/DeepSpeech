from spec_augment import *

if __name__ == "__main__":
    filename = "61-70968-0002.wav"
    log_mel_spectogram = get_log_mel_spectogram(filename)
    v, tau = log_mel_spectogram.shape[1], log_mel_spectogram.shape[2]
    with tf.Session() as sess:    
      augument = Augment(v, tau)
      input_spectogram, augment_spectogram = augument.execute()
      
      spectogram = sess.run(augment_spectogram, feed_dict = {input_spectogram : log_mel_spectogram})
      print(spectogram.shape)