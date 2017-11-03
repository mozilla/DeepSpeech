NODE_BUILD_TOOL ?= node-pre-gyp
NODE_ABI_TARGET ?= 
NODE_BUILD_VERBOSE ?= --verbose

include ../definitions.mk

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
	AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) CFLAGS="$(CFLAGS)" LDFLAGS="$(RPATH_NODEJS) $(LDFLAGS)" LIBS="$(LIBS)" $(NODE_BUILD_TOOL) $(NODE_PLATFORM_TARGET) $(NODE_ABI_TARGET) rebuild $(NODE_BUILD_VERBOSE)

copy-deps: build
	$(call copy_missing_libs,lib/binding/*/*/deepspeech.node,lib/)

package: copy-deps build
	$(NODE_BUILD_TOOL) $(NODE_PLATFORM_TARGET) $(NODE_ABI_TARGET) package $(NODE_BUILD_VERBOSE)

npm-pack: clean
	npm install node-pre-gyp@0.6.x
	npm pack $(NODE_BUILD_VERBOSE)

deepspeech_wrap.cxx: deepspeech.i
	swig -c++ -javascript -node deepspeech.i
