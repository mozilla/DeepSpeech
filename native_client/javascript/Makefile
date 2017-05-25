
default: bindings

clean:
	rm -f deepspeech_wrap.cxx
	rm -rf ./build

bindings: deepspeech_wrap.cxx
	node-gyp configure
	node-gyp build

deepspeech_wrap.cxx: deepspeech.i
	swig -c++ -javascript -node deepspeech.i
