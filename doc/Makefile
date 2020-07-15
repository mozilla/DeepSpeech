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

.PHONY: help pip3 Makefile

pip3:
	$(PIP_INSTALL) -r ../taskcluster/docs-requirements.txt

submodule:
	git submodule update --init --remote -- ../doc/examples

# Add submodule update dependency to Sphinx's "html" target
html: Makefile submodule pip3
	@PATH=$$HOME/.local/bin:`pwd`/../node_modules/.bin/:$$PATH \
	     $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

dist: html
	cd $(BUILDDIR)/html/ && zip -r9 ../../html.zip *

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile pip3
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
