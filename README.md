<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![Built Status](https://api.cirrus-ci.com/github/<USER>/classify_covid.svg?branch=main)](https://cirrus-ci.com/github/<USER>/classify_covid)
[![ReadTheDocs](https://readthedocs.org/projects/classify_covid/badge/?version=latest)](https://classify_covid.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/classify_covid/main.svg)](https://coveralls.io/r/<USER>/classify_covid)
[![PyPI-Server](https://img.shields.io/pypi/v/classify_covid.svg)](https://pypi.org/project/classify_covid/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/classify_covid.svg)](https://anaconda.org/conda-forge/classify_covid)
[![Monthly Downloads](https://pepy.tech/badge/classify_covid/month)](https://pepy.tech/project/classify_covid)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/classify_covid)
-->

# Covid Classification for Mobile Inference

> This repository contains code for modelling, integrations, tests, and documentation for covid classification

## VSCode Setup

In order to set up the necessary container:

1. Setup Github with SSH and Install Docker Desktop like mentioned in ML-Team-Guidelines
2. clone this repository and open the repo folder insider vscode
   ```bash
   keyboard command : left CTRL + Shift + P
   Type: ">Dev Containers: Rebuild Container"
   ```

## Installation

In order to set up the necessary environment:

1. review and uncomment what you need in `environment.yml` and create an environment `classify_covid` with the help of [conda]:
   ```bash
   conda env create -f requirements/<framework>/environment.yml
   e.g. conda env create -f requirements/tensorflow/environment.yml
   ```
2. activate the new environment with:
   ```bash
   conda activate <framework>
   e.g. conda activate tensorflow
   ```
   ```bash
   python -m ipykernel install --user --name=<framework>
   e.g. python -m ipykernel install --user --name=tensorflow
   ```

> **_NOTE:_**  The conda environment will have classify_covid installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.


Optional and needed only once after `git clone`:

3. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

4. to run tests manually you can use below function

   ```bash
   pre-commit run --all-files
   # we suggest to use this regularly in order to avoid errors
   ```

5. to start jupyterlab, you can use below command
   ```bash
   jupyter lab --no-browser --port=8890 --allow-root --ip='0.0.0.0' --NotebookApp.token='' --NotebookApp.password=''
   # In order to open jupyterlab, you can go to ports > globe icon beside local address link
   ```

<!--
4. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.

Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n classify_covid -f environment.lock.yml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
   ```
-->

## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── requirements
│   ├── tensorflow          <- contains tensorflow environment and requirements
│   │   └── environment.yml 
│   └── pytorch             <- contains pytorch environment and requirements
│       └── environment.yml 
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
│
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── classify_covid      <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.3.1 and the [dsproject extension] 0.7.2.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
