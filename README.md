# NCLE

Code to reproduce experiments from the paper: "Neural composite likelihood estimation: simulation based inference for time series".

We have included code for reproducing the two examples - AR(1) and GARCH(1,1) time series models - presented in the paper.

## Setup

For compatibility between packages, we recommend using `NumPy` version 1.26.4 and `sbi` version 0.22.0 or older.

## Usage

`train_*.py` files are used to train NCLE, `ci_*.py` files are used to obtain Godambe confidence intervals, and `log_probs_*.py` files are used to obtain conditional likelihoods. These were used to produce the plots in the paper.

To train NCLE on the GARCH(1,1) model, go to `Ex1-GARCH11` and run `train_nle.py` with the arguments specified by `sys.argv` in the relevant Python script controlling the NCLE settings. For example, to use 10,000 training samples and a batch length of 10, run:

```bash
python train_nle.py 10000 10
```
This outputs a pickle file (`.pkl`) in which the results are stored.

The other scripts follow a similar procedure. Note that different scripts accept different arguments, so be sure to check `sys.argv` in the relevant script before running it.
