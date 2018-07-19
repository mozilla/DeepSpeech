NODE_BUILD_TOOL    ?= node-pre-gyp
NODE_ABI_TARGET    ?= 
NODE_BUILD_VERBOSE ?= --verbose
PROJECT_NAME       ?= deepspeech
PROJECT_VERSION    ?= $(shell cat ../../VERSION | tr -d '\n')

include ../definitions.mk

default: build

clean:
	rm -f deepspeech_wrap.cxx package.json README.md
	rm -rf ./build/

clean-npm-pack:
	rm -fr ./node_modules/
	rm -fr ./deepspeech-*.tgz

really-clean: clean clean-npm-pack
	rm -fr ./lib/

README.md:
	cp ../../README.md README.md

package.json: package.json.in
	sed \
		-e 's/$$(PROJECT_NAME)/$(PROJECT_NAME)/' \
		-e 's/$$(PROJECT_VERSION)/$(PROJECT_VERSION)/' \
		package.json.in > package.json && cat package.json

configure: deepspeech_wrap.cxx package.json
	$(NODE_BUILD_TOOL) configure $(NODE_BUILD_VERBOSE)

build: configure deepspeech_wrap.cxx
	AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) CFLAGS="$(CFLAGS)" CXXFLAGS="$(CXXFLAGS)" LDFLAGS="$(RPATH_NODEJS) $(LDFLAGS)" LIBS="$(LIBS)" $(NODE_BUILD_TOOL) $(NODE_PLATFORM_TARGET) $(NODE_ABI_TARGET) rebuild $(NODE_BUILD_VERBOSE)

copy-deps: build
	$(call copy_missing_libs,lib/binding/*/*/*/deepspeech.node,lib/binding/*/*/)

node-wrapper: copy-deps build
	$(NODE_BUILD_TOOL) $(NODE_PLATFORM_TARGET) $(NODE_ABI_TARGET) package $(NODE_BUILD_VERBOSE)

npm-pack: clean package.json README.md
	npm install node-pre-gyp@0.6.x
	npm pack $(NODE_BUILD_VERBOSE)

deepspeech_wrap.cxx: deepspeech.i
	swig -c++ -javascript -node deepspeech.i
