
Building DeepSpeech Binaries
============================

If you'd like to build the DeepSpeech binaries yourself, you'll need the following pre-requisites downloaded and installed:


* `Mozilla's TensorFlow r2.2 branch <https://github.com/mozilla/tensorflow/tree/r2.2>`_
* `Bazel 2.0.0 <https://github.com/bazelbuild/bazel/releases/tag/2.0.0>`_
* `General TensorFlow requirements <https://www.tensorflow.org/install/install_sources>`_
* `libsox <https://sourceforge.net/projects/sox/>`_

It is required to use our fork of TensorFlow since it includes fixes for common problems encountered when building the native client files.

If you'd like to build the language bindings or the decoder package, you'll also need:


* `SWIG >= 3.0.12 <http://www.swig.org/>`_.
  Unfortunately, NodeJS / ElectronJS after 10.x support on SWIG is a bit behind, and while there are pending patches proposed to upstream, it is not yet merged.
  The proper prebuilt patched version (covering linux, windows and macOS) of SWIG should get installed under `native_client/ <native_client/>`_ as soon as you build any bindings that requires it.

* `node-pre-gyp <https://github.com/mapbox/node-pre-gyp>`_ (for Node.JS bindings only)

Dependencies
------------

If you follow these instructions, you should compile your own binaries of DeepSpeech (built on TensorFlow using Bazel).

For more information on configuring TensorFlow, read the docs up to the end of `"Configure the Build" <https://www.tensorflow.org/install/source#configure_the_build>`_.

TensorFlow: Clone & Checkout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone our fork of TensorFlow and checkout the correct version:

.. code-block::

   git clone https://github.com/mozilla/tensorflow.git
   git checkout origin/r2.2

Bazel: Download & Install
^^^^^^^^^^^^^^^^^^^^^^^^^

First, install Bazel 2.0.0 following the `Bazel installation documentation <https://docs.bazel.build/versions/2.0.0/install.html>`_.

TensorFlow: Configure with Bazel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After you have installed the correct version of Bazel, configure TensorFlow:

.. code-block::

   cd tensorflow
   ./configure

Compile DeepSpeech
------------------

Compile ``libdeepspeech.so``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within your TensorFlow checkout, create a symbolic link to the DeepSpeech ``native_client`` directory. Assuming DeepSpeech and TensorFlow checkouts are in the same directory, do:

.. code-block::

   cd tensorflow
   ln -s ../DeepSpeech/native_client ./

You can now use Bazel to build the main DeepSpeech library, ``libdeepspeech.so``\ . Add ``--config=cuda`` if you want a CUDA build.

.. code-block::

   bazel build --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //native_client:libdeepspeech.so

The generated binaries will be saved to ``bazel-bin/native_client/``.

Compile Language Bindings
^^^^^^^^^^^^^^^^^^^^^^^^^

Now, ``cd`` into the ``DeepSpeech/native_client`` directory and use the ``Makefile`` to build all the language bindings (C++ client, Python package, Nodejs package, etc.). Set the environment variable ``TFDIR`` to point to your TensorFlow checkout.

.. code-block::

   TFDIR=~/tensorflow
   cd ../DeepSpeech/native_client
   make deepspeech

Installing your own Binaries
----------------------------

After building, the library files and binary can optionally be installed to a system path for ease of development. This is also a required step for bindings generation.

.. code-block::

   PREFIX=/usr/local sudo make install

It is assumed that ``$PREFIX/lib`` is a valid library path, otherwise you may need to alter your environment.

Install Python bindings
^^^^^^^^^^^^^^^^^^^^^^^

Included are a set of generated Python bindings. After following the above build and installation instructions, these can be installed by executing the following commands (or equivalent on your system):

.. code-block::

   cd native_client/python
   make bindings
   pip install dist/deepspeech*

The API mirrors the C++ API and is demonstrated in `client.py <python/client.py>`_. Refer to `deepspeech.h <deepspeech.h>`_ for documentation.

Install NodeJS / ElectronJS bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After following the above build and installation instructions, the Node.JS bindings can be built:

.. code-block::

   cd native_client/javascript
   make build
   make npm-pack

This will create the package ``deepspeech-VERSION.tgz`` in ``native_client/javascript``.

Install the CTC decoder package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the ``ds_ctcdecoder`` package, you'll need the general requirements listed above (in particular SWIG). The command below builds the bindings using eight (8) processes for compilation. Adjust the parameter accordingly for more or less parallelism.

.. code-block::

   cd native_client/ctcdecode
   make bindings NUM_PROCESSES=8
   pip install dist/*.whl

Cross-building
--------------

RPi3 ARMv7 and LePotato ARM64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We do support cross-compilation. Please refer to our ``mozilla/tensorflow`` fork, where we define the following ``--config`` flags:


* ``--config=rpi3`` and ``--config=rpi3_opt`` for Raspbian / ARMv7
* ``--config=rpi3-armv8`` and ``--config=rpi3-armv8_opt`` for ARMBian / ARM64

So your command line for ``RPi3`` and ``ARMv7`` should look like:

.. code-block::

   bazel build --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --config=monolithic --config=rpi3 --config=rpi3_opt -c opt --copt=-O3 --copt=-fvisibility=hidden //native_client:libdeepspeech.so

And your command line for ``LePotato`` and ``ARM64`` should look like:

.. code-block::

   bazel build --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --config=monolithic --config=rpi3-armv8 --config=rpi3-armv8_opt -c opt --copt=-O3 --copt=-fvisibility=hidden //native_client:libdeepspeech.so

While we test only on RPi3 Raspbian Buster and LePotato ARMBian Buster, anything compatible with ``armv7-a cortex-a53`` or ``armv8-a cortex-a53`` should be fine.

The ``deepspeech`` binary can also be cross-built, with ``TARGET=rpi3`` or ``TARGET=rpi3-armv8``. This might require you to setup a system tree using the tool ``multistrap`` and the multitrap configuration files: ``native_client/multistrap_armbian64_buster.conf`` and ``native_client/multistrap_raspbian_buster.conf``.
The path of the system tree can be overridden from the default values defined in ``definitions.mk`` through the ``RASPBIAN`` ``make`` variable.

.. code-block::

   cd ../DeepSpeech/native_client
   make TARGET=<system> deepspeech

Android devices
^^^^^^^^^^^^^^^

We have preliminary support for Android relying on TensorFlow Lite, with Java and JNI bindinds. For more details on how to experiment with those, please refer to ``native_client/java/README.rst``.

Please refer to TensorFlow documentation on how to setup the environment to build for Android (SDK and NDK required).

You can build the ``libdeepspeech.so`` using (ARMv7):

.. code-block::

   bazel build --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --config=monolithic --config=android --config=android_arm --define=runtime=tflite --action_env ANDROID_NDK_API_LEVEL=21 --cxxopt=-std=c++11 --copt=-D_GLIBCXX_USE_C99 //native_client:libdeepspeech.so

Or (ARM64):

.. code-block::

   bazel build --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --config=monolithic --config=android --config=android_arm64 --define=runtime=tflite --action_env ANDROID_NDK_API_LEVEL=21 --cxxopt=-std=c++11 --copt=-D_GLIBCXX_USE_C99 //native_client:libdeepspeech.so

Building the ``deepspeech`` binary will happen through ``ndk-build`` (ARMv7):

.. code-block::

   cd ../DeepSpeech/native_client
   $ANDROID_NDK_HOME/ndk-build APP_PLATFORM=android-21 APP_BUILD_SCRIPT=$(pwd)/Android.mk NDK_PROJECT_PATH=$(pwd) APP_STL=c++_shared TFDIR=$(pwd)/../../tensorflow/ TARGET_ARCH_ABI=armeabi-v7a

And (ARM64):

.. code-block::

   cd ../DeepSpeech/native_client
   $ANDROID_NDK_HOME/ndk-build APP_PLATFORM=android-21 APP_BUILD_SCRIPT=$(pwd)/Android.mk NDK_PROJECT_PATH=$(pwd) APP_STL=c++_shared TFDIR=$(pwd)/../../tensorflowx/ TARGET_ARCH_ABI=arm64-v8a
