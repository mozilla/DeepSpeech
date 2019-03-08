# OpenFST port for Windows

OpenFst is a library for constructing, combining, optimizing, and searching
weighted finite-state transducers (FSTs), maintained by Google and released
under the [Apache 2.0 license](./COPYING). The home page for the library is
located at http://openfst.org/. Check the [original README file](./README)
for the current version, as we are not updating this file with the current
release version. Make sure also to check the [NEWS](./NEWS) file for the
latest changes.

## Releases

We track [original releases](http://www.openfst.org/twiki/bin/view/FST/FstDownload)
of the library, and try to keep [ours](https://github.com/kkm000/openfst/releases)
in step. Sometimes we may skip a release, but we strive for that to be rather an
exception. With each release, we drop a set of pre-built .exe files compiled for
the x64 architecture and optimized for execution speed. We use the latest
Microsoft compilers for it, so you may find you need to download and install the
latest Microsoft C runtime. See [Microsoft KB2977003](https://support.microsoft.com/help/2977003/)
for instructions.

We do not release pre-built libraries, however, because Microsoft compiler ABI
changes between versions, and is different for Debug and Release builds. You
must build a library matching your toolset on your own.

## Build

There are two build options: Visual Studio and CMake. We maintain a set of
build scripts for both. You will need a recent enough Visual Studio for either
build flavor. Microsoft provides an option to download the free Visual Studio
[Community Edition](https://visualstudio.microsoft.com/downloads/), which is
adequate.

### Visual Studio

Open `openfst.sln`, then read the comments in files under the "READ ME BEFORE
BUILD" solution folder. Generally, you may just hit Build and get the
libraries, unless you need fine-tuning, such as selecting a different toolset,
or want to build with MSBuild from command line. The solutions builds only
static libraries, with debug information embedded in C7 format for the
simplicity of use. Set the platform to `x86` or `x64` to build a respective
32- or 64-bit version of the library and tools.

The `bin` project builds multiple executable files by invoking itself
recursively once for each executable. All .vcxproj files have been scripted and
are maintained by hand. It takes a long time to build in Release mode. If you
only need the libfst library, build it alone from the Project Explorer.

All build outputs are placed into the `build_output` directory under the
solution root.

### CMake

Follow the normal CMake build procedure to generate build files. With CMake you
have an option of building dynamic libraries shared by the executables.

## Limitations

* Memory-mapped files are not supported (we may add the support in the future
  though), because it is very system-dependent. OpenFST supports reading e. g.
  CompactFST files into allocated memoty when memory mapping is not compiled in.
* Dynamic registration of arc and FST types is not supported in the Visual Studio
  project versions (as they build only static libraries). CMake build does not
  have this limitation. Due to ABI being specific to Microsoft compiler version,
  dynamically registered types must be compiled with strictly the same compiler
  of the same major version, and mostly same build flags. This is quite hard to
  get right, and is not recommended.

## Structure of the repository and tagging

The `original` branch contains only imported original OpenFST files, with one
exception of .gitignore file added. Tags of the form `orig/1.6.9.1` specify the
version and revision number of the library. Every commit on the `orignal` branch
contains the source URL for the tarball release of OpenFST that was committed.
The last version point corresponds to the revision of OpenFST version. Most (but
not all) of the versions has had only one revision, and therefore end in `.1`.
The `winport` branch contains the port, with corresponding tags of the form
`win/1.6.9.1`.

You can review the changes to source code only with the git command e. g.

`git diff orig/1.6.9.1..win/1.6.9.1 -- "*.cc" "*.h"`

The GitHub interface does not provide filtering by extension, so you will see
all CMakefiles and MSBuild files added, but it may be useful if you want to
examine changes in a particular file.

We try to keep changes to an absolute minimum. Most of them are due to
incompatibilities between the gcc and cl compilers, and only a minor portion
is due to the differences between the Linux and Windows platforms.

## Maintainers

The repository is maintained by [@kkm000](https://github.com/kkm000) and
[@jtrmal](https://github.com/jtrmal).

Open an issue to let us know if you discover a problem with the port. We react
to them promptly. Be sure to include a problem description, error text id any,
and the compiler or Visual Studio version (and CMake version, if you use it).
Do not hesitate to ask questions. We will try to help you.

Since we do not (by the definition of port) extend the OpenFST library itself,
please contact its authors with questions and suggestions related to the
original library. If in doubt, contact both teams.

Also let us know if you have a related development that you believe should be
linked to from this file.

---

_Copyright (c) 2008-current Google Inc._  
_Copyright (c) 2016-current SmartAction LLC (kkm)_  
_Copyright (c) 2016-current Johns Hopkins University (J. Trmal)_
