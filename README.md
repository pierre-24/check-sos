# check-sos

Let's figure out SOS... Once again ;)


## Overview

Based on perturbation theory, the sum-over-states (SOS) expression of Orr and Ward states that any component of any nonlinear optical tensor can be computed as a sum involving ground-to-excited and excited-to-excited state contributions ([details here](white-papers/SOS.pdf)). 
While straightforward to implement, this expression introduces two types of divergences: **secular divergences** and **resonances**. 
Resonances are intrinsic to perturbation theory, while secular divergences are mathematical artifacts that should be avoided.
This project provides a (non-efficient) implementation of non-divergent formulas, tested against the divergent versions. 

The input files used are in the format defined by MultiWFN (refer to Section 3.27.2.1 of the [MultiWFN manual](http://sobereva.com/multiwfn/misc/Multiwfn_3.8_dev.pdf)). 
These files can be generated using system models based on the VB-nCT theory, which also allows for validation of the results (see [this document](white-papers/few-states.pdf)).

## Installation

Install the package easily with:

```bash
pip install git+https://github.com/pierre-24/check-sos.git 
```

## Usage

In order to compute a nonlinear tensor using SOS, use:

```bash 
sos-get-tensor -i tests/2-state.txt
```

Options are:

+ `-i`: input file in the MultiWFN format. Use `--eV` if the excitation energies are give in eV.
+ `-w` for the laser frequency, in atomic unit (default is 0).
+ `-f` for a space-separated list of input field. (default is `-f "1 1"`) For example, to compute $\beta(-2\omega;\omega,\omega)$ (SHG phenomenon), use `-f "1 1"`. For $\gamma(-\omega;\omega,-\omega,\omega)$, use `-f "1 -1 1"`. The value of $\omega$ is given by `-w`.

To create an input file using a simple VB-nCT model, use:

```bash
sos-create-system tests/CT_dipoles.txt -t .05 -T .01 -m 0.2
```

Options are:

+ `-t`, `-T` and `-m`: values for $t$, $T$ and $m_{CT}$, as described in Section 2.3 of [this document](white-papers/few-states.pdf).
+ `--eV` to output energies in eV rather than atomic unit, which is the default.

It is possible to combine both commands:

```bash
sos-create-system tests/CT_dipoles.txt -t .05 -T .01 -m 0.2 | sos-get-tensor -f "-1 1"
```

... Which will compute the optical rectification tensor :)
