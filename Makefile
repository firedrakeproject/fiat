.PHONY: lint
lint: srclint doclint

# Adds file annotations to Github Actions (only useful on CI)
GITHUB_ACTIONS_FORMATTING=0
ifeq ($(GITHUB_ACTIONS_FORMATTING), 1)
	FLAKE8_FORMAT=--format='::error file=%(path)s,line=%(row)d,col=%(col)d,title=%(code)s::%(path)s:%(row)d:%(col)d: %(code)s %(text)s'
else
	FLAKE8_FORMAT=
endif

.PHONY: srclint
srclint:
	@echo "    Linting FIAT"
	@python -m flake8 $(FLAKE8_FORMAT) --statistics FIAT
	@echo "    Linting FInAT"
	@python -m flake8 $(FLAKE8_FORMAT) --statistics finat
	@echo "    Linting GEM"
	@python -m flake8 $(FLAKE8_FORMAT) --statistics gem
	@echo "    Linting tests"
	@python -m flake8 $(FLAKE8_FORMAT) --statistics test

.PHONY: doclint
doclint:
	@echo "    Checking FIAT docstring style"
	@python -m pydocstyle FIAT
	@echo "    Checking FInAT docstring style"
	@python -m pydocstyle finat
	@echo "    Checking GEM docstring style"
	@python -m pydocstyle gem
