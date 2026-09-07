# Top-level convenience Makefile. Most real targets live in subdirs.

# Single source of truth for the pin; the CI typecheck job runs this target.
PYRIGHT_VERSION := 1.1.410

.PHONY: docs html docs-clean docs-livehtml test typecheck

docs html:
	$(MAKE) -C docs html

docs-clean:
	$(MAKE) -C docs clean

docs-livehtml:
	$(MAKE) -C docs livehtml

test:
	uv run pytest

typecheck:
	uvx pyright@$(PYRIGHT_VERSION)
