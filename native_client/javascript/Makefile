NODE_BUILD_TOOL ?= node-pre-gyp
NODE_ABI_TARGET ?= 
NODE_BUILD_VERBOSE ?= --verbose

default: build

clean:
	rm -f deepspeech_wrap.cxx
	rm -rf ./build/

clean-npm-pack:
	rm -fr ./node_modules/
	rm -fr ./deepspeech-*.tgz

really-clean: clean clean-npm-pack
	rm -fr ./lib/

configure: deepspeech_wrap.cxx
	$(NODE_BUILD_TOOL) configure $(NODE_BUILD_VERBOSE)

build: configure deepspeech_wrap.cxx
	$(NODE_BUILD_TOOL) $(NODE_ABI_TARGET) rebuild $(NODE_BUILD_VERBOSE)

package: build
	LDFLAGS="-L$(TFDIR)/bazel-bin/tensorflow -L$(TFDIR)/bazel-bin/native_client" $(NODE_BUILD_TOOL) $(NODE_ABI_TARGET) package $(NODE_BUILD_VERBOSE)

npm-pack: clean
	npm install node-pre-gyp@0.6.x
	npm pack $(NODE_BUILD_VERBOSE)

deepspeech_wrap.cxx: deepspeech.i
	swig -c++ -javascript -node deepspeech.i
