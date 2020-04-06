# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = DeepSpeech
SOURCEDIR     = .
BUILDDIR      = .build

PIP_INSTALL   ?= pip3 install --user

NPM_ROOT           ?= $(shell npm root)

NODE_MODULES_BIN   ?= $(NPM_ROOT)/.bin/
ifeq ($(findstring _NT,$(OS)),_NT)
# On Windows, we seem to need both in PATH for node-pre-gyp as well as tsc
# they do not get installed the same way.
NODE_MODULES_BIN   := $(shell cygpath -u $(NPM_ROOT)/.bin/):$(shell cygpath -u `dirname "$(NPM_ROOT)"`)
endif

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help pip3 npm Makefile doxygen-c doxygen-java

doxygen-c:
	cd ../ && doxygen doc/doxygen-c.conf

doxygen-java:
	cd ../ && doxygen doc/doxygen-java.conf

doxygen-dotnet:
	cd ../ && doxygen doc/doxygen-dotnet.conf

pip3:
	$(PIP_INSTALL) -r ../taskcluster/docs-requirements.txt

npm:
	npm install typedoc@0.17.4 typescript@3.8.3 typedoc@0.17.4 electron@^1.7.9 node-gyp@5.x @types/argparse@1.0.x @types/node@13.9.x

submodule:
	git submodule update --init --remote

# Add submodule update dependency to Sphinx's "html" target
html: Makefile submodule pip3 npm doxygen-c doxygen-java doxygen-dotnet
	PATH="$(NODE_MODULES_BIN):${PATH}" $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

dist: html
	cd $(BUILDDIR)/html/ && zip -r9 ../../html.zip *

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile pip3 npm doxygen-c doxygen-java doxygen-dotnet
	PATH="$(NODE_MODULES_BIN):${PATH}" $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
