
DeepSpeech Java / Android bindings
==================================

This is still preliminary work. Please refer to ``native_client/README.rst`` for
building ``libdeepspeech.so`` and ``deepspeech`` binary for Android on ARMv7 and
ARM64 arch.

Android Java / JNI bindings: ``libdeepspeech``
==================================================

Java / JNI bindings are available under the ``libdeepspeech`` subdirectory.
Building depends on prebuilt shared object.  Please ensure to place
``libdeepspeech.so`` into the ``libdeepspeech/libs/{arm64-v8a,armeabi-v7a}/``
matching subdirectories.

Building the bindings is managed by ``gradle`` and should be limited to issuing
``./gradlew libdeepspeech:build``\ , producing an ``AAR`` package in
``./libdeepspeech/build/outputs/aar/``. This can later be used by other
Gradle-based build with the following configuration:

.. code-block::

   implementation 'deepspeech.mozilla.org:libdeepspeech:VERSION@aar'

Please note that you might have to copy the file to a local Maven repository
and adapt file naming (when missing, the error message should states what
filename it expects and where).

Android demo APK
================

Provided is a very simple Android demo app that allows you to test the library.
You can build it with ``make apk`` and install the resulting APK file. Please
refer to Gradle documentation for more details.

The ``APK`` should be produced in ``/app/build/outputs/apk/``. This demo app might
require external storage permissions. You can then push models files to your
device, set the path to the file in the UI and try to run on an audio file.
When running, it should first play the audio file and then run the decoding. At
the end of the decoding, you should be presented with the decoded text as well
as time elapsed to decode in miliseconds.

Running ``deepspeech`` via adb
==================================

You should use ``adb push`` to send data to device, please refer to Android
documentation on how to use that.

Please push DeepSpeech data to ``/sdcard/deepspeech/``\ , including:


* ``output_graph.tflite`` which is the TF Lite model
* ``lm.binary`` and ``trie`` files, if you want to use the language model ; please
  be aware that too big language model will make the device run out of memory

Then, push binaries from ``native_client.tar.xz`` to ``/data/local/tmp/ds``\ :


* ``deepspeech``
* ``libdeepspeech.so``
* ``libc++_shared.so``

You should then be able to run as usual, using a shell from ``adb shell``\ :

.. code-block::

   user@device$ cd /data/local/tmp/ds/
   user@device$ LD_LIBRARY_PATH=$(pwd)/ ./deepspeech [...]

Please note that Android linker does not support ``rpath`` so you have to set
``LD_LIBRARY_PATH``. Properly wrapped / packaged bindings does embed the library
at a place the linker knows where to search, so Android apps will be fine.
