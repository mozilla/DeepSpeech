# Building DeepSpeech native client for Windows

Now we can build the native client of DeepSpeech and run inference on Windows using the C# client, to do that we need to compile the `native_client`.

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Getting the code](#getting-the-code)
- [Configuring the paths](#configuring-the-paths)
- [Adding environment variables](#adding-environment-variables)
    - [MSYS2 paths](#msys2-paths)
    - [BAZEL path](#bazel-path)
    - [Python path](#python-path)
    - [CUDA paths](#cuda-paths)
- [Building the native_client](#building-the-native_client)
    - [Build for CPU](#cpu)
    - [Build with CUDA support](#gpu-with-cuda)
- [Using the generated library](#using-the-generated-library)

## Prerequisites

* Windows 10
* [Windows 10 SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk)
* [Visual Studio 2017 Community](https://visualstudio.microsoft.com/vs/community/) 
* [Git Large File Storage](https://git-lfs.github.com/)
* [TensorFlow Windows pre-requisites](https://www.tensorflow.org/install/source_windows)

Inside the Visual Studio Installer enable `MS Build Tools` and `VC++ 2015.3 v14.00 (v140) toolset for desktop`.

If you want to enable CUDA support you need to follow the steps in [the TensorFlow docs for building on Windows with CUDA](https://www.tensorflow.org/install/gpu#windows_setup).

We highly recommend sticking to the recommended versions of CUDA/cuDNN in order to avoid compilation errors caused by incompatible versions. We only test with the versions recommended by TensorFlow.

## Getting the code

We need to clone `mozilla/DeepSpeech` and `mozilla/tensorflow`.

```bash
git clone https://github.com/mozilla/DeepSpeech
```

```bash
git clone --branch r1.13 https://github.com/mozilla/tensorflow
```

## Configuring the paths

We need to create a symbolic link, for this example let's suppose that we cloned into `D:\cloned` and now the structure looks like:

    .
    ├── D:\
    │   ├── cloned                 # Contains DeepSpeech and tensorflow side by side
    │   │   ├── DeepSpeech         # Root of the cloned DeepSpeech
    │   │   ├── tensorflow         # Root of the cloned Mozilla's tensorflow 
    └── ...

Change your path accordingly to your path structure, for the structure above we are going to use the following command:

```bash
mklink /d "D:\cloned\tensorflow\native_client" "D:\cloned\DeepSpeech\native_client"
```

## Adding environment variables

After you have installed the requirements there are few environment variables that we need to add to our `PATH` variable of the system variables.

#### MSYS2 paths

For MSYS2 we need to add `bin` directory, if you installed in the default route the path that we need to add should looks like `C:\msys64\usr\bin`. Now we can run `pacman`:

```bash
pacman -Syu
pacman -Su
pacman -S patch unzip
```

#### BAZEL path

For BAZEL we need to add the path to the executable, make sure you rename the executable to `bazel`.

To check the version installed you can run:

```bash
bazel version
```

#### PYTHON path

Add your `python.exe` path to the `PATH` variable.


#### CUDA paths

If you run CUDA enabled `native_client` we need to add the following to the `PATH` variable.

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
```

### Building the native_client

There's one last command to run before building, you need to run the [configure.py](https://github.com/mozilla/tensorflow/blob/master/configure.py) inside `tensorflow` cloned directory.

At this point we are ready to start building the `native_client`, go to `tensorflow` directory that you cloned, following our examples should be `D:\cloned\tensorflow`.  

#### CPU
We will add AVX/AVX2 support in the command, please make sure that your CPU supports these instructions before adding the flags, if not you can remove them.

```bash
bazel build -c opt --copt=/arch:AVX --copt=/arch:AVX2 //native_client:libdeepspeech.so
```

#### GPU with CUDA
If you enabled CUDA in [configure.py](https://github.com/mozilla/tensorflow/blob/master/configure.py) configuration command now you can add `--config=cuda` to compile with CUDA support.

```bash
bazel build -c opt --config=cuda --copt=/arch:AVX --copt=/arch:AVX2 //native_client:libdeepspeech.so
```

Be patient, if you enabled AVX/AVX2 and CUDA it will take a long time. Finally you should see it stops and shows the path to the generated `libdeepspeech.so`.


## Using the generated library

As for now we can only use the generated `libdeepspeech.so` with the C# clients, go to [native_client/dotnet/](https://github.com/mozilla/DeepSpeech/tree/master/native_client/dotnet) in your DeepSpeech directory and open the Visual Studio solution, then we need to build in debug or release mode, finally we just need to copy `libdeepspeech.so` to the generated `x64/Debug` or `x64/Release` directory.
