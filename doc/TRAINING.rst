Training Your Own Model
=======================

Prerequisites for training a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `Python 3.6 <https://www.python.org/>`_
* `Git Large File Storage <https://git-lfs.github.com/>`_
* Mac or Linux environment

Getting the training code
^^^^^^^^^^^^^^^^^^^^^^^^^

Install `Git Large File Storage <https://git-lfs.github.com/>`_ either manually or through a package-manager if available on your system. Then clone the DeepSpeech repository normally:

.. code-block:: bash

   git clone https://github.com/mozilla/DeepSpeech

Creating a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In creating a virtual environment you will create a directory containing a ``python3`` binary and everything needed to run deepspeech. You can use whatever directory you want. For the purpose of the documentation, we will rely on ``$HOME/tmp/deepspeech-train-venv``. You can create it using this command:

.. code-block::

   $ python3 -m venv $HOME/tmp/deepspeech-train-venv/

Once this command completes successfully, the environment will be ready to be activated.

Activating the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each time you need to work with DeepSpeech, you have to *activate* this virtual environment. This is done with this simple command:

.. code-block::

   $ source $HOME/tmp/deepspeech-train-venv/bin/activate

Installing Python dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the required dependencies using ``pip3``\ :

.. code-block:: bash

   cd DeepSpeech
   pip3 install -e .

The ``webrtcvad`` Python package might require you to ensure you have proper tooling to build Python modules:

.. code-block:: bash

   sudo apt-get install python3-dev

You'll also need to install the ``ds_ctcdecoder`` Python package. ``ds_ctcdecoder`` is required for decoding the outputs of the ``deepspeech`` acoustic model into text. You can use ``util/taskcluster.py`` with the ``--decoder`` flag to get a URL to a binary of the decoder package appropriate for your platform and Python version:

.. code-block:: bash

   pip3 install $(python3 util/taskcluster.py --decoder)

This command will download and install the ``ds_ctcdecoder`` package. You can override the platform with ``--arch`` if you want the package for ARM7 (\ ``--arch arm``\ ) or ARM64 (\ ``--arch arm64``\ ). If you prefer building the ``ds_ctcdecoder`` package from source, see the :github:`native_client README file <native_client/README.rst>`.

Recommendations
^^^^^^^^^^^^^^^

If you have a capable (NVIDIA, at least 8GB of VRAM) GPU, it is highly recommended to install TensorFlow with GPU support. Training will be significantly faster than using the CPU. To enable GPU support, you can do:

.. code-block:: bash

   pip3 uninstall tensorflow
   pip3 install 'tensorflow-gpu==1.15.2'

Please ensure you have the required `CUDA dependency <USING.rst#cuda-dependency>`_.

It has been reported for some people failure at training:

.. code-block::

   tensorflow.python.framework.errors_impl.UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
        [[{{node tower_0/conv1d/Conv2D}}]]

Setting the ``TF_FORCE_GPU_ALLOW_GROWTH`` environment variable to ``true`` seems to help in such cases. This could also be due to an incorrect version of libcudnn. Double check your versions with the `TensorFlow 1.15 documentation <USING.rst#cuda-dependency>`_.

Common Voice training data
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Common Voice corpus consists of voice samples that were donated through Mozilla's `Common Voice <https://voice.mozilla.org/>`_ Initiative.
You can download individual CommonVoice v2.0 language data sets from `here <https://voice.mozilla.org/data>`_.
After extraction of such a data set, you'll find the following contents:


* the ``*.tsv`` files output by CorporaCreator for the downloaded language
* the mp3 audio files they reference in a ``clips`` sub-directory.

For bringing this data into a form that DeepSpeech understands, you have to run the CommonVoice v2.0 importer (\ ``bin/import_cv2.py``\ ):

.. code-block:: bash

   bin/import_cv2.py --filter_alphabet path/to/some/alphabet.txt /path/to/extracted/language/archive

Providing a filter alphabet is optional. It will exclude all samples whose transcripts contain characters not in the specified alphabet. 
Running the importer with ``-h`` will show you some additional options.

Once the import is done, the ``clips`` sub-directory will contain for each required ``.mp3`` an additional ``.wav`` file.
It will also add the following ``.csv`` files:


* ``clips/train.csv``
* ``clips/dev.csv``
* ``clips/test.csv``

All entries in these CSV files refer to their samples by absolute paths. So moving this sub-directory would require another import or tweaking the CSV files accordingly.

To use Common Voice data during training, validation and testing, you pass (comma separated combinations of) their filenames into ``--train_files``\ , ``--dev_files``\ , ``--test_files`` parameters of ``DeepSpeech.py``.

If, for example, Common Voice language ``en`` was extracted to ``../data/CV/en/``\ , ``DeepSpeech.py`` could be called like this:

.. code-block:: bash

   ./DeepSpeech.py --train_files ../data/CV/en/clips/train.csv --dev_files ../data/CV/en/clips/dev.csv --test_files ../data/CV/en/clips/test.csv

Training a model
^^^^^^^^^^^^^^^^

The central (Python) script is ``DeepSpeech.py`` in the project's root directory. For its list of command line options, you can call:

.. code-block:: bash

   ./DeepSpeech.py --helpfull

To get the output of this in a slightly better-formatted way, you can also look up the option definitions in :github:`util/flags.py <util/flags.py>`.

For executing pre-configured training scenarios, there is a collection of convenience scripts in the ``bin`` folder. Most of them are named after the corpora they are configured for. Keep in mind that most speech corpora are *very large*, on the order of tens of gigabytes, and some aren't free. Downloading and preprocessing them can take a very long time, and training on them without a fast GPU (GTX 10 series or newer recommended) takes even longer.

**If you experience GPU OOM errors while training, try reducing the batch size with the ``--train_batch_size``\ , ``--dev_batch_size`` and ``--test_batch_size`` parameters.**

As a simple first example you can open a terminal, change to the directory of the DeepSpeech checkout, activate the virtualenv created above, and run:

.. code-block:: bash

   ./bin/run-ldc93s1.sh

This script will train on a small sample dataset composed of just a single audio file, the sample file for the `TIMIT Acoustic-Phonetic Continuous Speech Corpus <https://catalog.ldc.upenn.edu/LDC93S1>`_, which can be overfitted on a GPU in a few minutes for demonstration purposes. From here, you can alter any variables with regards to what dataset is used, how many training iterations are run and the default values of the network parameters.

Feel also free to pass additional (or overriding) ``DeepSpeech.py`` parameters to these scripts. Then, just run the script to train the modified network.

Each dataset has a corresponding importer script in ``bin/`` that can be used to download (if it's freely available) and preprocess the dataset. See ``bin/import_librivox.py`` for an example of how to import and preprocess a large dataset for training with DeepSpeech.

If you've run the old importers (in ``util/importers/``\ ), they could have removed source files that are needed for the new importers to run. In that case, simply remove the extracted folders and let the importer extract and process the dataset from scratch, and things should work.

Training with automatic mixed precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Automatic Mixed Precision (AMP) training on GPU for TensorFlow has been recently [introduced](https://medium.com/tensorflow/automatic-mixed-precision-in-tensorflow-for-faster-ai-training-on-nvidia-gpus-6033234b2540).

Mixed precision training makes use of both FP32 and FP16 precisions where appropriate. FP16 operations can leverage the Tensor cores on NVIDIA GPUs (Volta, Turing or newer architectures) for improved throughput. Mixed precision training also often allows larger batch sizes. DeepSpeech GPU automatic mixed precision training can be enabled via the flag value `--auto_mixed_precision=True`.

```
DeepSpeech.py --train_files ./train.csv --dev_files ./dev.csv --test_files ./test.csv --automatic_mixed_precision=True
```

On a Volta generation V100 GPU, automatic mixed precision speeds up DeepSpeech training and evaluation by ~30%-40%.

Checkpointing
^^^^^^^^^^^^^

During training of a model so-called checkpoints will get stored on disk. This takes place at a configurable time interval. The purpose of checkpoints is to allow interruption (also in the case of some unexpected failure) and later continuation of training without losing hours of training time. Resuming from checkpoints happens automatically by just (re)starting training with the same ``--checkpoint_dir`` of the former run. Alternatively, you can specify more fine grained options with ``--load_checkpoint_dir`` and ``--save_checkpoint_dir``, which specify separate locations to use for loading and saving checkpoints respectively. If not specified these flags use the same value as ``--checkpoint_dir``, ie. load from and save to the same directory.

Be aware however that checkpoints are only valid for the same model geometry they had been generated from. In other words: If there are error messages of certain ``Tensors`` having incompatible dimensions, this is most likely due to an incompatible model change. One usual way out would be to wipe all checkpoint files in the checkpoint directory or changing it before starting the training.

Exporting a model for inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the ``--export_dir`` parameter is provided, a model will have been exported to this directory during training.
Refer to the corresponding :github:`README.rst <native_client/README.rst>` for information on building and running a client that can use the exported model.

Exporting a model for TFLite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to experiment with the TF Lite engine, you need to export a model that is compatible with it, then use the ``--export_tflite`` flags. If you already have a trained model, you can re-export it for TFLite by running ``DeepSpeech.py`` again and specifying the same ``checkpoint_dir`` that you used for training, as well as passing ``--export_tflite --export_dir /model/export/destination``.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you'd like to use one of the pre-trained models released by Mozilla to bootstrap your training process (transfer learning, fine tuning), you can do so by using the ``--checkpoint_dir`` flag in ``DeepSpeech.py``. Specify the path where you downloaded the checkpoint from the release, and training will resume from the pre-trained model.

For example, if you want to fine tune the entire graph using your own data in ``my-train.csv``\ , ``my-dev.csv`` and ``my-test.csv``\ , for three epochs, you can something like the following, tuning the hyperparameters as needed:

.. code-block:: bash

   mkdir fine_tuning_checkpoints
   python3 DeepSpeech.py --n_hidden 2048 --checkpoint_dir path/to/checkpoint/folder --epochs 3 --train_files my-train.csv --dev_files my-dev.csv --test_files my_dev.csv --learning_rate 0.0001

Note: the released models were trained with ``--n_hidden 2048``\ , so you need to use that same value when initializing from the release models. Since v0.6.0, the release models are also trained with ``--train_cudnn``\ , so you'll need to specify that as well. If you don't have a CUDA compatible GPU, then you can workaround it by using the ``--load_cudnn`` flag. Use ``--helpfull`` to get more information on how the flags work. If you try to load a release model without following these steps, you'll get an error similar to this:

.. code-block::

   Key cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/bias/Adam not found in checkpoint

UTF-8 mode
^^^^^^^^^^

DeepSpeech includes a UTF-8 operating mode which can be useful to model languages with very large alphabets, such as Chinese Mandarin. For details on how it works and how to use it, see :ref:`decoder-docs`.

Training with augmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Augmentation is a useful technique for better generalization of machine learning models. Thus, a pre-processing pipeline with various augmentation techniques on raw pcm and spectrogram has been implemented and can be used while training the model. Following are the available augmentation techniques that can be enabled at training time by using the corresponding flags in the command line.

Audio Augmentation
~~~~~~~~~~~~~~~~~~


#. **Standard deviation for Gaussian additive noise:** ``--data_aug_features_additive``
#. **Standard deviation for Normal distribution around 1 for multiplicative noise:** ``--data_aug_features_multiplicative`` 
#. **Standard deviation for speeding-up tempo. If Standard deviation is 0, this augmentation is not performed:** ``--augmentation_speed_up_std`` 

Spectrogram Augmentation
~~~~~~~~~~~~~~~~~~~~~~~~

Inspired by Google Paper on `SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition <https://arxiv.org/abs/1904.08779>`_


#. 
   **Keep rate of dropout augmentation on a spectrogram (if 1, no dropout will be performed on the spectrogram)**\ : 


   * Keep Rate : ``--augmentation_spec_dropout_keeprate value between range [0 - 1]`` 

#. 
   **Whether to use frequency and time masking augmentation:** 


   * Enable / Disable : ``--augmentation_freq_and_time_masking / --noaugmentation_freq_and_time_masking``  
   * Max range of masks in the frequency domain when performing freqtime-mask augmentation: ``--augmentation_freq_and_time_masking_freq_mask_range eg: 5``
   * Number of masks in the frequency domain when performing freqtime-mask augmentation: ``--augmentation_freq_and_time_masking_number_freq_masks eg: 3`` 
   * Max range of masks in the time domain when performing freqtime-mask augmentation: ``--augmentation_freq_and_time_masking_time_mask_rangee eg: 2`` 
   * Number of masks in the time domain when performing freqtime-mask augmentation: ``augmentation_freq_and_time_masking_number_time_masks eg: 3`` 

#. 
   **Whether to use spectrogram speed and tempo scaling:** 


   * Enable / Disable : ``--augmentation_pitch_and_tempo_scaling / --noaugmentation_pitch_and_tempo_scaling.``  
   * Min value of pitch scaling: ``--augmentation_pitch_and_tempo_scaling_min_pitch eg:0.95`` 
   * Max value of pitch scaling: ``--augmentation_pitch_and_tempo_scaling_max_pitch eg:1.2``  
   * Max value of tempo scaling: ``--augmentation_pitch_and_tempo_scaling_max_tempo eg:1.2``  

