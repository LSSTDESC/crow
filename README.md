# CROW
**C**luster **R**econstruction of **O**bservables **W**orkbench: **CROW**

The LSST-DESC Cluster Reconstruction of Observables Workbench (CROW) code is a DESC tool consisting of a Python library for predicting galaxy cluster observabless.

## Table of contents
1. [Installing CROW](#installing)
2. [Using CROW](#using)
3. [Contributing to CROW](#contributing)
5. [Contact](#contact)

# Installing Crow <a name="installing"></a>

Crow can be installed with `pip` or `conda`.

For a `pip` installation, run:

```bash
    pip install lsstdesc-crow
```

For a `conda` installation, run:

```bash
    conda install -c conda-forge lsstdesc-crow
```
After, to use is in your code, just do 

```bash
import crow
```

## Requirements <a name="requirements"></a>

Crow requires Python version 3.11 or later.

### Dependencies <a name="dependencies"></a>

Crow has the following dependencies:

- [NumPy](https://www.numpy.org/) (v2 or later)
- [SciPy](https://scipy.org/) (v1.12 or later)
- [Pyccl](https://ccl.readthedocs.io/en/latest/index.html)
- [CLMM](https://lsstdesc.org/CLMM/)
- [NumCosmo](https://numcosmo.readthedocs.io/en/latest/)

# Using Crow <a name="using"></a>

This code has been released by DESC, although it is still under active
development. You are welcome to re-use the code, which is open source and available under
terms consistent with our
[LICENSE](https://github.com/LSSTDESC/crow/blob/main/LICENSE) ([BSD
3-Clause](https://opensource.org/licenses/BSD-3-Clause)).

Example usage can be found in the `notebooks` folder.

**DESC Projects**: External contributors and DESC members wishing to
use Crow for DESC projects should consult with the DESC Clusters analysis
working group (CL WG) conveners, ideally before the work has started, but
definitely before any publication or posting of the work to the arXiv.

**Non-DESC Projects by DESC members**: If you are in the DESC
community, but planning to use Crow in a non-DESC project, it would be
good practice to contact the CL WG co-conveners and/or the Crow
Team leads as well (see Contact section).  A desired outcome would be for your
non-DESC project concept and progress to be presented to the working group,
so working group members can help co-identify tools and/or ongoing development
that might mutually benefit your non-DESC project and ongoing DESC projects.

**External Projects by Non-DESC members**: If you are not from the DESC
community, you are also welcome to contact Crow Team leads to introduce
your project and share feedback.


# Contributing to Crow <a name="contributing"></a>

You are welcome to contribute to the code. To do so, please make sure
you use `isort` and `black` on your code and assure you provide unit tests.

## Updating Public Documentation on lsstdesc.org <a name="updating_public_docs"></a>

This is easy! Once you have merged all approved changes into `main`, you will want to update the public documentation.
Just go to the `publish-docs` branch (`git checkout publish-docs`) and run the `./publish_docs` script.

# Contact <a name="contact"></a>

If you have comments, questions, or feedback, please contact the current leads
 of the LSST DESC Crow Team: Michel Aguena
(m-aguena, aguena@inaf.it) and Eduardo Barroso (eduardojsbarroso,
barroso@lapp.in2p3.fr)

