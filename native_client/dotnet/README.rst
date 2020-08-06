
Building Mozilla Voice STT native client for Windows
=============================================

Now we can build the native client of Mozilla Voice STT and run inference on Windows using the C# client, to do that we need to compile the ``native_client``.

**Table of Contents**


* `Prerequisites <#prerequisites>`_
* `Getting the code <#getting-the-code>`_
* `Configuring the paths <#configuring-the-paths>`_
* `Adding environment variables <#adding-environment-variables>`_

  * `MSYS2 paths <#msys2-paths>`_
  * `BAZEL path <#bazel-path>`_
  * `Python path <#python-path>`_
  * `CUDA paths <#cuda-paths>`_

* `Building the native_client <#building-the-native_client>`_

  * `Build for CPU <#cpu>`_
  * `Build with CUDA support <#gpu-with-cuda>`_

* `Using the generated library <#using-the-generated-library>`_

Prerequisites
-------------


* Windows 10
* `Windows 10 SDK <https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk>`_
* `Visual Studio 2019 Community <https://visualstudio.microsoft.com/vs/community/>`_ 
* `TensorFlow Windows pre-requisites <https://www.tensorflow.org/install/source_windows>`_

Inside the Visual Studio Installer enable ``MS Build Tools`` and ``VC++ 2019 v16.00 (v160) toolset for desktop``.

If you want to enable CUDA support you need to follow the steps in `the TensorFlow docs for building on Windows with CUDA <https://www.tensorflow.org/install/gpu#windows_setup>`_.

We highly recommend sticking to the recommended versions of CUDA/cuDNN in order to avoid compilation errors caused by incompatible versions. We only test with the versions recommended by TensorFlow.

Getting the code
----------------

We need to clone ``mozilla/DeepSpeech``.

.. code-block:: bash

   git clone https://github.com/mozilla/DeepSpeech
   git submodule sync tensorflow/
   git submodule update --init tensorflow/

Configuring the paths
---------------------

There should already be a symbolic link, for this example let's suppose that we cloned into ``D:\cloned`` and now the structure looks like:

.. code-block::

   .
   ├── D:\
   │   ├── cloned                 # Contains Mozilla Voice STT and tensorflow side by side
   │   │   └── DeepSpeech         # Root of the cloned Mozilla Voice STT
   │   │       ├── tensorflow     # Root of the cloned Mozilla's tensorflow 
   └── ...


Change your path accordingly to your path structure, for the structure above we are going to use the following command if the symbolic link does not exists:

.. code-block:: bash

   mklink /d "D:\cloned\DeepSpeech\tensorflow\native_client" "D:\cloned\DeepSpeech\native_client"

Adding environment variables
----------------------------

After you have installed the requirements there are few environment variables that we need to add to our ``PATH`` variable of the system variables.

MSYS2 paths
~~~~~~~~~~~

For MSYS2 we need to add ``bin`` directory, if you installed in the default route the path that we need to add should looks like ``C:\msys64\usr\bin``. Now we can run ``pacman``:

.. code-block:: bash

   pacman -Syu
   pacman -Su
   pacman -S patch unzip

BAZEL path
~~~~~~~~~~

For BAZEL we need to add the path to the executable, make sure you rename the executable to ``bazel``.

To check the version installed you can run:

.. code-block:: bash

   bazel version

PYTHON path
~~~~~~~~~~~

Add your ``python.exe`` path to the ``PATH`` variable.

CUDA paths
~~~~~~~~~~

If you run CUDA enabled ``native_client`` we need to add the following to the ``PATH`` variable.

.. code-block::

   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin

Building the native_client
^^^^^^^^^^^^^^^^^^^^^^^^^^

There's one last command to run before building, you need to run the `configure.py <https://github.com/mozilla/tensorflow/blob/master/configure.py>`_ inside ``tensorflow`` cloned directory.

At this point we are ready to start building the ``native_client``, go to ``tensorflow`` sub-directory, following our examples should be ``D:\cloned\DeepSpeech\tensorflow``.  

CPU
~~~

We will add AVX/AVX2 support in the command, please make sure that your CPU supports these instructions before adding the flags, if not you can remove them.

.. code-block:: bash

   bazel build --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" -c opt --copt=/arch:AVX --copt=/arch:AVX2 //native_client:libmozilla_voice_stt.so

GPU with CUDA
~~~~~~~~~~~~~

If you enabled CUDA in `configure.py <https://github.com/mozilla/tensorflow/blob/master/configure.py>`_ configuration command now you can add ``--config=cuda`` to compile with CUDA support.

.. code-block:: bash

   bazel build --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" -c opt --config=cuda --copt=/arch:AVX --copt=/arch:AVX2 //native_client:libmozilla_voice_stt.so

Be patient, if you enabled AVX/AVX2 and CUDA it will take a long time. Finally you should see it stops and shows the path to the generated ``libmozilla_voice_stt.so``.

Using the generated library
---------------------------

As for now we can only use the generated ``libmozilla_voice_stt.so`` with the C# clients, go to `native_client/dotnet/ <https://github.com/mozilla/DeepSpeech/tree/master/native_client/dotnet>`_ in your Mozilla Voice STT directory and open the Visual Studio solution, then we need to build in debug or release mode, finally we just need to copy ``libmozilla_voice_stt.so`` to the generated ``x64/Debug`` or ``x64/Release`` directory.
