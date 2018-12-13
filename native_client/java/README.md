DeepSpeech Java / Android bindings
==================================

This is still preliminary work. Please refer to `native_client/README.md` for
building `libdeepspeech.so` and `deepspeech` binary for Android on ARMv7 and
ARM64 arch.

Running `deepspeech` via adb
============================
You should use `adb push` to send data to device, please refer to Android
documentation on how to use that.

Please push DeepSpeech data to `/sdcard/deepspeech/`, including:
 - `output_graph.tflite` which is the TF Lite model
 - `alphabet.txt`
 - `lm.binary` and `trie` files, if you want to use the language model ; please
   be aware that too big language model will make the device run out of memory

Then, push binaries from `native_client.tar.xz` to `/data/local/tmp/ds`:
 - `deepspeech`
 - `libdeepspeech.so`
 - `libc++_shared.so`

You should then be able to run as usual, using a shell from `adb shell`:
```
user@device$ cd /data/local/tmp/ds/
user@device$ LD_LIBRARY_PATH=$(pwd)/ ./deepspeech [...]
```

Please note that Android linker does not support `rpath` so you have to set
`LD_LIBRARY_PATH`. Properly wrapped / packaged bindings does embed the library
at a place the linker knows where to search, so Android apps will be fine.
