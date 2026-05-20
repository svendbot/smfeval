# Minimal Makefile for Sphinx documentation. Run from inside `docs/`:
#
#   nix develop --command bash -c "cd docs && make html"
#
# Or from the repo root:
#
#   nix develop --command make -C docs html

SPHINXBUILD ?= uv run --group docs sphinx-build
SPHINXOPTS  ?= -W
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help html clean livehtml

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)" _autosummary

# Catch-all target: route anything else to sphinx-build's -M dispatcher
# (e.g. `make linkcheck`, `make epub`, etc.).
%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
