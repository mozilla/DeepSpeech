# Java bindings
Full project description and documentation on GitHub: [https://github.com/mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech).

## Android bindings
For use with Android

### Preparation
1. Rename `./build.gradle.android` to `build.gradle`
2. Rename `./libdeepspeech/build.gradle.android` to `build.gradle` 
3. Rename `./libdeepspeech/CMakeLists_android.txt` to `CMakeLists.txt`

### Build
1. In `./` run `make`

>Note: The current example app in `./App` is not up to date with the latest changes to the bindings!

## Standalone Java Bindings for DeepSpeech
For use with standalone Java

See [standalone.md](standalone.md)