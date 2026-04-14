SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = CROW
SOURCEDIR     = .
BUILDDIR      = _build
RM            = rm -rf
SPHINXAPIDOC  = sphinx-apidoc
MODULEDIR     = ../crow

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

.PHONY: html
html: sphinx-apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(0) -W

.PHONY: clean
clean:
	@$(RM) _build/* api/ api.rst compiled-examples/ index.rst

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -W

sphinx-apidoc:
	@$(SPHINXAPIDOC) --separate --no-toc -f -M -o api $(MODULEDIR)
