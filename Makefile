JULIA := julia

.PHONY: install docs test readme

all: install

install:
	$(JULIA) -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'

test:
	$(JULIA) --project=. -e 'using Pkg; Pkg.test()'

docs:
	$(JULIA) --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
	$(JULIA) --project=docs/ docs/make.jl

readme:
	$(JULIA) --project=readme/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
	quarto render readme/README.qmd --output README.md
	mv readme/README_files README_files
