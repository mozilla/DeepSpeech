# Project DeepSpeech

Project DeepSpeech is an open source Speech-To-Text engine that uses a model trained by machine learning techniques, based on [Baidu's Deep Speech research paper](https://arxiv.org/abs/1412.5567). Project DeepSpeech uses Google's [TensorFlow](https://www.tensorflow.org/) project to facilitate implementation.

## Prerequisites

* [TensorFlow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#download-and-setup)
* [IronPython](http://ironpython.net/download/)
* [SciPy](http://scipy.org/install.html)
* [PyXDG](https://pypi.python.org/pypi/pyxdg)
* [python_speech_features](https://pypi.python.org/pypi/python_speech_features)
* [python sox] (https://pypi.python.org/pypi/sox)

## Recommendations

If you have a capable (nVidia) GPU, it is highly recommended to install TensorFlow with GPU support. Training will likely be significantly quicker than using the CPU.

## Training a model

Open a terminal, change to the directory of the DeepSpeech checkout and run `ipython notebook DeepSpeech.ipynb`. This should open your default browser with the DeepSpeech notebook. From here, you can inspect the notebook and alter any variables with regards to what dataset is used, how many training iterations are run and the default values of the training parameters.

Once you are satisfied with the settings, you can begin to train a model by selecting 'Cell' from the notebook menu bar and choosing 'Run All'.

## Exporting a model for serving

If the `ds_export_dir` environment variable is set, or the `export_dir` variable is set manually, a model will have been exported to this directory during training. If training has been performed without exporting a model, a model can be exported by setting the variable to the directory you'd like to export to (e.g. `export_dir = os.path.join(checkpoint_dir, 'export')`) and running the model exporting cell manually. If the notebook has been restarted since training, you will need to run all the cells above the training cell first before running the export cell, to declare and initialise the required variables and functions.

Refer to the corresponding [README.md](client/README.md) for information on building and running the client.
