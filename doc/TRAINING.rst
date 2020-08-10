.. _training-docs:

Training Your Own Model
=======================

Prerequisites for training a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Python 3.6 <https://www.python.org/>`_
* Mac or Linux environment

Getting the training code
^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the Mozilla Voice STT repository:

.. code-block:: bash

   git clone https://github.com/mozilla/DeepSpeech

Creating a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In creating a virtual environment you will create a directory containing a ``python3`` binary and everything needed to run Mozilla Voice STT. You can use whatever directory you want. For the purpose of the documentation, we will rely on ``$HOME/tmp/stt-train-venv``. You can create it using this command:

.. code-block::

   $ python3 -m venv $HOME/tmp/stt-train-venv/

Once this command completes successfully, the environment will be ready to be activated.

Activating the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each time you need to work with Mozilla Voice STT, you have to *activate* this virtual environment. This is done with this simple command:

.. code-block::

   $ source $HOME/tmp/stt-train-venv/bin/activate

Installing Mozilla Voice STT Training Code and its dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the required dependencies using ``pip3``\ :

.. code-block:: bash

   cd DeepSpeech
   pip3 install --upgrade pip==20.0.2 wheel==0.34.2 setuptools==46.1.3
   pip3 install --upgrade -e .

Remember to re-run the last ``pip3 install`` command above when you update the training code (for example by pulling new changes), in order to update any dependencies.

The ``webrtcvad`` Python package might require you to ensure you have proper tooling to build Python modules:

.. code-block:: bash

   sudo apt-get install python3-dev

Recommendations
^^^^^^^^^^^^^^^

If you have a capable (NVIDIA, at least 8GB of VRAM) GPU, it is highly recommended to install TensorFlow with GPU support. Training will be significantly faster than using the CPU. To enable GPU support, you can do:

.. code-block:: bash

   pip3 uninstall tensorflow
   pip3 install 'tensorflow-gpu==1.15.2'

Please ensure you have the required :ref:`CUDA dependency <cuda-deps>`.

It has been reported for some people failure at training:

.. code-block::

   tensorflow.python.framework.errors_impl.UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
        [[{{node tower_0/conv1d/Conv2D}}]]

Setting the ``TF_FORCE_GPU_ALLOW_GROWTH`` environment variable to ``true`` seems to help in such cases. This could also be due to an incorrect version of libcudnn. Double check your versions with the :ref:`TensorFlow 1.15 documentation <cuda-deps>`.

Basic Dockerfile for training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide ``Dockerfile.train`` to automatically set up a basic training environment in Docker. You need to generate the Dockerfile from the template using:
This should ensure that you'll re-use the upstream Python 3 TensorFlow GPU-enabled Docker image.

.. code-block:: bash

   make Dockerfile.train

If you want to specify a different Mozilla Voice STT repository / branch, you can pass ``MOZILLA_VOICE_STT_REPO`` or ``MOZILLA_VOICE_STT_SHA`` parameters:

.. code-block:: bash

   make Dockerfile.train MOZILLA_VOICE_STT_REPO=git://your/fork MOZILLA_VOICE_STT_SHA=origin/your-branch

Common Voice training data
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Common Voice corpus consists of voice samples that were donated through Mozilla's `Common Voice <https://voice.mozilla.org/>`_ Initiative.
You can download individual CommonVoice v2.0 language data sets from `here <https://voice.mozilla.org/data>`_.
After extraction of such a data set, you'll find the following contents:


* the ``*.tsv`` files output by CorporaCreator for the downloaded language
* the mp3 audio files they reference in a ``clips`` sub-directory.

For bringing this data into a form that Mozilla Voice STT understands, you have to run the CommonVoice v2.0 importer (\ ``bin/import_cv2.py``\ ):

.. code-block:: bash

   bin/import_cv2.py --filter_alphabet path/to/some/alphabet.txt /path/to/extracted/language/archive

Providing a filter alphabet is optional. It will exclude all samples whose transcripts contain characters not in the specified alphabet. 
Running the importer with ``-h`` will show you some additional options.

Once the import is done, the ``clips`` sub-directory will contain for each required ``.mp3`` an additional ``.wav`` file.
It will also add the following ``.csv`` files:


* ``clips/train.csv``
* ``clips/dev.csv``
* ``clips/test.csv``

Entries in CSV files can refer to samples by their absolute or relative paths. Here, the importer produces relative paths.

To use Common Voice data during training, validation and testing, you pass (comma separated combinations of) their filenames into ``--train_files``\ , ``--dev_files``\ , ``--test_files`` parameters of ``DeepSpeech.py``.

If, for example, Common Voice language ``en`` was extracted to ``../data/CV/en/``\ , ``DeepSpeech.py`` could be called like this:

.. code-block:: bash

   python3 DeepSpeech.py --train_files ../data/CV/en/clips/train.csv --dev_files ../data/CV/en/clips/dev.csv --test_files ../data/CV/en/clips/test.csv

Training a model
^^^^^^^^^^^^^^^^

The central (Python) script is ``DeepSpeech.py`` in the project's root directory. For its list of command line options, you can call:

.. code-block:: bash

   python3 DeepSpeech.py --helpfull

To get the output of this in a slightly better-formatted way, you can also look at the flag definitions in :ref:`training-flags`.

For executing pre-configured training scenarios, there is a collection of convenience scripts in the ``bin`` folder. Most of them are named after the corpora they are configured for. Keep in mind that most speech corpora are *very large*, on the order of tens of gigabytes, and some aren't free. Downloading and preprocessing them can take a very long time, and training on them without a fast GPU (GTX 10 series or newer recommended) takes even longer.

**If you experience GPU OOM errors while training, try reducing the batch size with the ``--train_batch_size``\ , ``--dev_batch_size`` and ``--test_batch_size`` parameters.**

As a simple first example you can open a terminal, change to the directory of the Mozilla Voice STT checkout, activate the virtualenv created above, and run:

.. code-block:: bash

   ./bin/run-ldc93s1.sh

This script will train on a small sample dataset composed of just a single audio file, the sample file for the `TIMIT Acoustic-Phonetic Continuous Speech Corpus <https://catalog.ldc.upenn.edu/LDC93S1>`_, which can be overfitted on a GPU in a few minutes for demonstration purposes. From here, you can alter any variables with regards to what dataset is used, how many training iterations are run and the default values of the network parameters.

Feel also free to pass additional (or overriding) ``DeepSpeech.py`` parameters to these scripts. Then, just run the script to train the modified network.

Each dataset has a corresponding importer script in ``bin/`` that can be used to download (if it's freely available) and preprocess the dataset. See ``bin/import_librivox.py`` for an example of how to import and preprocess a large dataset for training with Mozilla Voice STT.

Some importers might require additional code to properly handled your locale-specific requirements. Such handling is dealt with ``--validate_label_locale`` flag that allows you to source out-of-tree Python script that defines a ``validate_label`` function. Please refer to ``util/importers.py`` for implementation example of that function.
If you don't provide this argument, the default ``validate_label`` function will be used. This one is only intended for English language, so you might have consistency issues in your data for other languages.

For example, in order to use a custom validation function that disallows any sample with "a" in its transcript, and lower cases everything else, you could put the following code in a file called ``my_validation.py`` and then use ``--validate_label_locale my_validation.py``:

.. code-block:: python

  def validate_label(label):
      if 'a' in label: # disallow labels with 'a'
          return None
      return label.lower() # lower case valid labels

If you've run the old importers (in ``util/importers/``\ ), they could have removed source files that are needed for the new importers to run. In that case, simply remove the extracted folders and let the importer extract and process the dataset from scratch, and things should work.

Training with automatic mixed precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Automatic Mixed Precision (AMP) training on GPU for TensorFlow has been recently [introduced](https://medium.com/tensorflow/automatic-mixed-precision-in-tensorflow-for-faster-ai-training-on-nvidia-gpus-6033234b2540).

Mixed precision training makes use of both FP32 and FP16 precisions where appropriate. FP16 operations can leverage the Tensor cores on NVIDIA GPUs (Volta, Turing or newer architectures) for improved throughput. Mixed precision training also often allows larger batch sizes. Automatic mixed precision training can be enabled by including the flag `--automatic_mixed_precision` at training time:

```
python3 DeepSpeech.py --train_files ./train.csv --dev_files ./dev.csv --test_files ./test.csv --automatic_mixed_precision
```

On a Volta generation V100 GPU, automatic mixed precision speeds up Mozilla Voice STT training and evaluation by ~30%-40%.

Checkpointing
^^^^^^^^^^^^^

During training of a model so-called checkpoints will get stored on disk. This takes place at a configurable time interval. The purpose of checkpoints is to allow interruption (also in the case of some unexpected failure) and later continuation of training without losing hours of training time. Resuming from checkpoints happens automatically by just (re)starting training with the same ``--checkpoint_dir`` of the former run. Alternatively, you can specify more fine grained options with ``--load_checkpoint_dir`` and ``--save_checkpoint_dir``, which specify separate locations to use for loading and saving checkpoints respectively. If not specified these flags use the same value as ``--checkpoint_dir``, ie. load from and save to the same directory.

Be aware however that checkpoints are only valid for the same model geometry they had been generated from. In other words: If there are error messages of certain ``Tensors`` having incompatible dimensions, this is most likely due to an incompatible model change. One usual way out would be to wipe all checkpoint files in the checkpoint directory or changing it before starting the training.

Exporting a model for inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the ``--export_dir`` parameter is provided, a model will have been exported to this directory during training.
Refer to the :ref:`usage instructions <usage-docs>` for information on running a client that can use the exported model.

Exporting a model for TFLite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to experiment with the TF Lite engine, you need to export a model that is compatible with it, then use the ``--export_tflite`` flags. If you already have a trained model, you can re-export it for TFLite by running ``DeepSpeech.py`` again and specifying the same ``checkpoint_dir`` that you used for training, as well as passing ``--export_tflite --export_dir /model/export/destination``. If you changed the alphabet you also need to add the ``--alphabet_config_path my-new-language-alphabet.txt`` flag.

Making a mmap-able model for inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``output_graph.pb`` model file generated in the above step will be loaded in memory to be dealt with when running inference.
This will result in extra loading time and memory consumption. One way to avoid this is to directly read data from the disk.

TensorFlow has tooling to achieve this: it requires building the target ``//tensorflow/contrib/util:convert_graphdef_memmapped_format`` (binaries are produced by our TaskCluster for some systems including Linux/amd64 and macOS/amd64), use ``util/taskcluster.py`` tool to download:

.. code-block::

   $ python3 util/taskcluster.py --source tensorflow --artifact convert_graphdef_memmapped_format --branch r1.15 --target .

Producing a mmap-able model is as simple as:

.. code-block::

   $ convert_graphdef_memmapped_format --in_graph=output_graph.pb --out_graph=output_graph.pbmm

Upon sucessfull run, it should report about conversion of a non-zero number of nodes. If it reports converting ``0`` nodes, something is wrong: make sure your model is a frozen one, and that you have not applied any incompatible changes (this includes ``quantize_weights``\ ).

Continuing training from a release model
----------------------------------------
There are currently two supported approaches to make use of a pre-trained Mozilla Voice STT model: fine-tuning or transfer-learning. Choosing which one to use is a simple decision, and it depends on your target dataset. Does your data use the same alphabet as the release model? If "Yes": fine-tune. If "No" use transfer-learning.

If your own data uses the *extact* same alphabet as the English release model (i.e. `a-z` plus `'`) then the release model's output layer will match your data, and you can just fine-tune the existing parameters. However, if you want to use a new alphabet (e.g. Cyrillic `а`, `б`, `д`), the output layer of a release Mozilla Voice STT model will *not* match your data. In this case, you should use transfer-learning (i.e. remove the trained model's output layer, and reinitialize a new output layer that matches your target character set.

N.B. - If you have access to a pre-trained model which uses UTF-8 bytes at the output layer you can always fine-tune, because any alphabet should be encodable as UTF-8.

.. _training-fine-tuning:

Fine-Tuning (same alphabet)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you'd like to use one of the pre-trained models released by Mozilla to bootstrap your training process (fine tuning), you can do so by using the ``--checkpoint_dir`` flag in ``DeepSpeech.py``. Specify the path where you downloaded the checkpoint from the release, and training will resume from the pre-trained model.

For example, if you want to fine tune the entire graph using your own data in ``my-train.csv``\ , ``my-dev.csv`` and ``my-test.csv``\ , for three epochs, you can something like the following, tuning the hyperparameters as needed:

.. code-block:: bash

   mkdir fine_tuning_checkpoints
   python3 DeepSpeech.py --n_hidden 2048 --checkpoint_dir path/to/checkpoint/folder --epochs 3 --train_files my-train.csv --dev_files my-dev.csv --test_files my_dev.csv --learning_rate 0.0001

Notes about the release checkpoints: the released models were trained with ``--n_hidden 2048``\ , so you need to use that same value when initializing from the release models. Since v0.6.0, the release models are also trained with ``--train_cudnn``\ , so you'll need to specify that as well. If you don't have a CUDA compatible GPU, then you can workaround it by using the ``--load_cudnn`` flag. Use ``--helpfull`` to get more information on how the flags work.

You also cannot use ```--automatic_mixed_precision``` when loading release checkpoints, as they do not use automatic mixed precision training.

If you try to load a release model without following these steps, you'll get an error similar to this:

.. code-block::

   E Tried to load a CuDNN RNN checkpoint but there were more missing variables than just the Adam moment tensors.


Transfer-Learning (new alphabet)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to continue training an alphabet-based Mozilla Voice STT model (i.e. not a UTF-8 model) on a new language, or if you just want to add new characters to your custom alphabet, you will probably want to use transfer-learning instead of fine-tuning. If you're starting with a pre-trained UTF-8 model -- even if your data comes from a different language or uses a different alphabet -- the model will be able to predict your new transcripts, and you should use fine-tuning instead.

In a nutshell, Mozilla Voice STT's transfer-learning allows you to remove certain layers from a pre-trained model, initialize new layers for your target data, stitch together the old and new layers, and update all layers via gradient descent. You will remove the pre-trained output layer (and optionally more layers) and reinitialize parameters to fit your target alphabet. The simplest case of transfer-learning is when you remove just the output layer.

In Mozilla Voice STT's implementation of transfer-learning, all removed layers will be contiguous, starting from the output layer. The key flag you will want to experiment with is ``--drop_source_layers``. This flag accepts an integer from ``1`` to ``5`` and allows you to specify how many layers you want to remove from the pre-trained model. For example, if you supplied ``--drop_source_layers 3``, you will drop the last three layers of the pre-trained model: the output layer, penultimate layer, and LSTM layer. All dropped layers will be reinintialized, and (crucially) the output layer will be defined to match your supplied target alphabet.

You need to specify the location of the pre-trained model with ``--load_checkpoint_dir`` and define where your new model checkpoints will be saved with ``--save_checkpoint_dir``. You need to specify how many layers to remove (aka "drop") from the pre-trained model: ``--drop_source_layers``. You also need to supply your new alphabet file using the standard ``--alphabet_config_path`` (remember, using a new alphabet is the whole reason you want to use transfer-learning).

.. code-block:: bash

       python3 DeepSpeech.py \
           --drop_source_layers 1 \
           --alphabet_config_path my-new-language-alphabet.txt \
           --save_checkpoint_dir path/to/output-checkpoint/folder \
           --load_checkpoint_dir path/to/release-checkpoint/folder \
           --train_files   my-new-language-train.csv \
           --dev_files   my-new-language-dev.csv \
           --test_files  my-new-language-test.csv

UTF-8 mode
^^^^^^^^^^

Mozilla Voice STT includes a UTF-8 operating mode which can be useful to model languages with very large alphabets, such as Chinese Mandarin. For details on how it works and how to use it, see :ref:`decoder-docs`.

.. _training-data-augmentation:

Augmentation
^^^^^^^^^^^^

Augmentation is a useful technique for better generalization of machine learning models. Thus, a pre-processing pipeline with various augmentation techniques on raw pcm and spectrogram has been implemented and can be used while training the model. Following are the available augmentation techniques that can be enabled at training time by using the corresponding flags in the command line.

Each sample of the training data will get treated by every specified augmentation in their given order. However: whether an augmentation will actually get applied to a sample is decided by chance on base of the augmentation's probability value. For example a value of ``p=0.1`` would apply the according augmentation to just 10% of all samples. This also means that augmentations are not mutually exclusive on a per-sample basis.

The ``--augment`` flag uses a common syntax for all augmentation types:

.. code-block::

  --augment augmentation_type1[param1=value1,param2=value2,...] --augment augmentation_type2[param1=value1,param2=value2,...] ...

For example, for the ``overlay`` augmentation:

.. code-block::

  python3 DeepSpeech.py --augment overlay[p=0.1,source=/path/to/audio.sdb,snr=20.0] ...


In the documentation below, whenever a value is specified as ``<float-range>`` or ``<int-range>``, it supports one of the follow formats:

  * ``<value>``: A constant (int or float) value.

  * ``<value>~<r>``: A center value with a randomization radius around it. E.g. ``1.2~0.4`` will result in picking of a uniformly random value between 0.8 and 1.6 on each sample augmentation.

  * ``<start>:<end>``: The value will range from `<start>` at the beginning of the training to `<end>` at the end of the training. E.g. ``-0.2:1.2`` (float) or ``2000:4000`` (int)

  * ``<start>:<end>~<r>``: Combination of the two previous cases with a ranging center value. E.g. ``4-6~2`` would at the beginning of the training pick values between 2 and 6 and at the end of the training between 4 and 8.

Ranges specified with integer limits will only assume integer (rounded) values.

.. warning::
    When feature caching is enabled, by default the cache has no expiration limit and will be used for the entire training run. This will cause these augmentations to only be performed once during the first epoch and the result will be reused for subsequent epochs. This would not only hinder value ranges from reaching their intended final values, but could also lead to unintended over-fitting. In this case flag ``--cache_for_epochs N`` (with N > 1) should be used to periodically invalidate the cache after every N epochs and thus allow samples to be re-augmented in new ways and with current range-values.

Every augmentation targets a certain representation of the sample - in this documentation these representations are referred to as *domains*.
Augmentations are applied in the following order:

1. **sample** domain: The sample just got loaded and its waveform is represented as a NumPy array. For implementation reasons these augmentations are the only ones that can be "simulated" through ``bin/play.py``.

2. **signal** domain: The sample waveform is represented as a tensor.

3. **spectrogram** domain: The sample spectrogram is represented as a tensor.

4. **features** domain: The sample's mel spectrogram features are represented as a tensor.

Within a single domain, augmentations are applied in the same order as they appear in the command-line.


Sample domain augmentations
---------------------------

**Overlay augmentation** ``--augment overlay[p=<float>,source=<str>,snr=<float-range>,layers=<int-range>]``
  Layers another audio source (multiple times) onto augmented samples.

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **source**: path to the sample collection to use for augmenting (\*.sdb or \*.csv file). It will be repeated if there are not enough samples left.

  * **snr**: signal to noise ratio in dB - positive values for lowering volume of the overlay in relation to the sample

  * **layers**: number of layers added onto the sample (e.g. 10 layers of speech to get "cocktail-party effect"). A layer is just a sample of the same duration as the sample to augment. It gets stitched together from as many source samples as required.


**Reverb augmentation** ``--augment reverb[p=<float>,delay=<float-range>,decay=<float-range>]``
  Adds simplified (no all-pass filters) `Schroeder reverberation <https://ccrma.stanford.edu/~jos/pasp/Schroeder_Reverberators.html>`_ to the augmented samples.

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **delay**: time delay in ms for the first signal reflection - higher values are widening the perceived "room"

  * **decay**: sound decay in dB per reflection - higher values will result in a less reflective perceived "room"


**Resample augmentation** ``--augment resample[p=<float>,rate=<int-range>]``
  Resamples augmented samples to another sample rate and then resamples back to the original sample rate.

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **rate**: sample-rate to re-sample to


**Codec augmentation** ``--augment codec[p=<float>,bitrate=<int-range>]``
  Compresses and then decompresses augmented samples using the lossy Opus audio codec.

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **bitrate**: bitrate used during compression


**Volume augmentation** ``--augment volume[p=<float>,dbfs=<float-range>]``
  Measures and levels augmented samples to a target dBFS value.

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **dbfs** : target volume in dBFS (default value of 3.0103 will normalize min and max amplitudes to -1.0/1.0)

Spectrogram domain augmentations
--------------------------------

**Pitch augmentation** ``--augment pitch[p=<float>,pitch=<float-range>]``
  Scales spectrogram on frequency axis and thus changes pitch.

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **pitch**: pitch factor by with the frequency axis is scaled (e.g. a value of 2.0 will raise audio frequency by one octave)


**Tempo augmentation** ``--augment tempo[p=<float>,factor=<float-range>]``
  Scales spectrogram on time axis and thus changes playback tempo.

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **factor**: speed factor by which the time axis is stretched or shrunken (e.g. a value of 2.0 will double playback tempo)


**Warp augmentation** ``--augment warp[p=<float>,nt=<int-range>,nf=<int-range>,wt=<float-range>,wf=<float-range>]``
  Applies a non-linear image warp to the spectrogram. This is achieved by randomly shifting a grid of equally distributed warp points along time and frequency axis.

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **nt**: number of equally distributed warp grid lines along time axis of the spectrogram (excluding the edges)

  * **nf**: number of equally distributed warp grid lines along frequency axis of the spectrogram (excluding the edges)

  * **wt**: standard deviation of the random shift applied to warp points along time axis (0.0 = no warp, 1.0 = half the distance to the neighbour point)

  * **wf**: standard deviation of the random shift applied to warp points along frequency axis (0.0 = no warp, 1.0 = half the distance to the neighbour point)


**Frequency mask augmentation** ``--augment frequency_mask[p=<float>,n=<int-range>,size=<int-range>]``
  Sets frequency-intervals within the augmented samples to zero (silence) at random frequencies. See the SpecAugment paper for more details - https://arxiv.org/abs/1904.08779

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **n**: number of intervals to mask

  * **size**: number of frequency bands to mask per interval

Multi domain augmentations
--------------------------

**Time mask augmentation** ``--augment time_mask[p=<float>,n=<int-range>,size=<float-range>,domain=<domain>]``
  Sets time-intervals within the augmented samples to zero (silence) at random positions.

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **n**: number of intervals to set to zero

  * **size**: duration of intervals in ms

  * **domain**: data representation to apply augmentation to - "signal", "features" or "spectrogram" (default)


**Dropout augmentation** ``--augment dropout[p=<float>,rate=<float-range>,domain=<domain>]``
  Zeros random data points of the targeted data representation.

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **rate**: dropout rate ranging from 0.0 for no dropout to 1.0 for 100% dropout

  * **domain**: data representation to apply augmentation to - "signal", "features" or "spectrogram" (default)


**Add augmentation** ``--augment add[p=<float>,stddev=<float-range>,domain=<domain>]``
  Adds random values picked from a normal distribution (with a mean of 0.0) to all data points of the targeted data representation.

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **stddev**: standard deviation of the normal distribution to pick values from

  * **domain**: data representation to apply augmentation to - "signal", "features" (default) or "spectrogram"


**Multiply augmentation** ``--augment multiply[p=<float>,stddev=<float-range>,domain=<domain>]``
  Multiplies all data points of the targeted data representation with random values picked from a normal distribution (with a mean of 1.0).

  * **p**: probability value between 0.0 (never) and 1.0 (always) if a given sample gets augmented by this method

  * **stddev**: standard deviation of the normal distribution to pick values from

  * **domain**: data representation to apply augmentation to - "signal", "features" (default) or "spectrogram"


Example training with all augmentations:

.. code-block:: bash

        python -u DeepSpeech.py \
          --train_files "train.sdb" \
          --feature_cache ./feature.cache \
          --cache_for_epochs 10 \
          --epochs 100 \
          --augment overlay[p=0.5,source=noise.sdb,layers=1,snr=50:20~10] \
          --augment reverb[p=0.1,delay=50.0~30.0,decay=10.0:2.0~1.0] \
          --augment resample[p=0.1,rate=12000:8000~4000] \
          --augment codec[p=0.1,bitrate=48000:16000] \
          --augment volume[p=0.1,dbfs=-10:-40] \
          --augment pitch[p=0.1,pitch=1~0.2] \
          --augment tempo[p=0.1,factor=1~0.5] \
          --augment warp[p=0.1,nt=4,nf=1,wt=0.5:1.0,wf=0.1:0.2] \
          --augment frequency_mask[p=0.1,n=1:3,size=1:5] \
          --augment time_mask[p=0.1,domain=signal,n=3:10~2,size=50:100~40] \
          --augment dropout[p=0.1,rate=0.05] \
          --augment add[p=0.1,domain=signal,stddev=0~0.5] \
          --augment multiply[p=0.1,domain=features,stddev=0~0.5] \
          [...]


The ``bin/play.py`` and ``bin/data_set_tool.py`` tools also support ``--augment`` parameters (for sample domain augmentations) and can be used for experimenting with different configurations or creating augmented data sets.

Example of playing all samples with reverberation and maximized volume:

.. code-block:: bash

        bin/play.py --augment reverb[p=0.1,delay=50.0,decay=2.0] --augment volume --random test.sdb

Example simulation of the codec augmentation of a wav-file first at the beginning and then at the end of an epoch:

.. code-block:: bash

        bin/play.py --augment codec[p=0.1,bitrate=48000:16000] --clock 0.0 test.wav
        bin/play.py --augment codec[p=0.1,bitrate=48000:16000] --clock 1.0 test.wav

Example of creating a pre-augmented test set:

.. code-block:: bash

        bin/data_set_tool.py \
          --augment overlay[source=noise.sdb,layers=1,snr=20~10] \
          --augment resample[rate=12000:8000~4000] \
          test.sdb test-augmented.sdb
