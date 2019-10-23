# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = DeepSpeech
SOURCEDIR     = .
BUILDDIR      = .build

PIP_INSTALL   ?= pip3 install --user

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
	npm install jsdoc@3.6.3

dist: html
	cd $(BUILDDIR)/html/ && zip -r9 ../../html.zip *

dist: html
	@cd $(BUILDDIR)/html/ && zip -r9 ../../html.zip *

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile pip3 npm doxygen-c doxygen-java doxygen-dotnet
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
