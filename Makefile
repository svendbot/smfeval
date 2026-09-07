# Top-level convenience Makefile. Most real targets live in subdirs.

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
	pyright
