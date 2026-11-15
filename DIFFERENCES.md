# What differs from hmmlearn 0.3.3, and how equality is checked

hmm-craft aims to fit every model exactly as hmmlearn 0.3.3 does. "Exactly"
means every fitted array, the score and the decode equal as bytes, not close.
This file lists the places where the two packages differ, then describes the
check that finds them.

## Differences

| Item | What changes | Option |
|---|---|---|
| Spherical covariances | `GaussianHMM(covariance_type="spherical").covars_` has one matrix per state, as documented; 0.3.3 returned one per state and feature. `sample()` draws each state with its own variance; in a fitted model 0.3.3 drew state `k` with the variance of state `k // n_features`, which is right with one feature and, with at least as many features as states, uses the variance of the first state for every state. No fit moves, and the stored variances are unchanged. | none |
| MAP decode | `decode(algorithm="map")` returns the sum of the log of the posterior probability of each chosen state. 0.3.3 returned the sum of the probabilities, a number between 0 and the number of samples. The states are unchanged. | none |
| `normalize(a, axis=0)` | A slice that sums to zero gives zeros. 0.3.3 gave NaN and a `RuntimeWarning`. | none |
| Invalid `lengths` | `lengths` must be a one-dimensional array of positive integers: an empty or two-dimensional array, a single integer, or an entry that is zero, negative, a float or a boolean raises `ValueError` in `fit` (before its start is computed), `score`, `score_samples`, `decode`, `predict` and `predict_proba`. 0.3.3 flattened a two-dimensional array and read a single integer as one sequence (the variational models' `fit` raised `TypeError` on both), counted a boolean entry as 1 and raised `TypeError` for a float. It accepted zero and negative entries when the total matched. `numpy.split` reads a negative entry as an index from the end, so some split `X` differently; a zero entry, and some negative ones, gave the kernel a sequence without samples, and the kernel then indexed outside its arrays: `score` and `decode` returned wrong values with no error, `score` with `implementation="scaling"` raised an underflow error, and `fit` raised. The kernel's functions now also raise `ValueError` for a matrix with no rows or no states. Valid `lengths` give the same results. | none |
| Covariances without means | A `GaussianHMM` with `"c"` but not `"m"` in `params` accumulates the state mass it needs, which 0.3.3 did not. With the default `covars_weight=1` and `means_weight=0`, variances (`covars_prior` + the posterior-weighted sum of squares) / 1e-5, for `spherical` averaged over the features: a division by the 1e-5 floor with no posterior mass. For `full` and `tied`, infinite covariances after the first M-step and `ValueError` from the second iteration, when `covars_weight` is at most `n_features`. `scripts/probe_covariance_only_fit.py` checks each clause on hmmlearn 0.3.3 built from its tag (Python 3.12.3, one thread; numpy 1.26.0, scipy 1.11.2, scikit-learn 1.3.1, threadpoolctl 3.8.0, and numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, threadpoolctl 3.8.0). This is the one change to a fit that has no option. | none |
| `state_occupancy_` | New attribute after `fit`, for every model: the expected number of samples in each state, from the last E-step. | none (new) |
| `covars_floor` | New `GaussianHMM` parameter. A positive number floors the covariances after every M-step that updates them: variances (`"spherical"`, `"diag"`) and eigenvalues (`"full"`, `"tied"`) below it are raised to it. It is independent of `min_covar`, which keeps its one effect, the term added to the covariance that starts the fit. | `covars_floor=None` (default): 0.3.3's fit |

A model pickled by hmmlearn 0.3.3 loads with `covars_floor=None`, and for a
spherical model stores the same variances as before.

## How equality is checked

`scripts/bit_equality.py` fits a fixed list of models with hmmlearn 0.3.3 and
with this package, each in its own environment, and compares the two sets of
results byte for byte. `scripts/test_bit_equality.py` tests the comparison
itself: it must fail on a one-ulp change, a flipped state, a changed shape or
a missing key, and each exception below must excuse its own difference and
nothing else.

* **Reference.** hmmlearn 0.3.3 built from its tag with the same compiler and
  build settings as this package, so a difference comes from the change and not
  from the build.
* **Environments.** Two environments with identical Python, numpy, scipy,
  scikit-learn and threadpoolctl versions; `compare` refuses to compare
  otherwise.
* **One thread.** `dump` refuses to run unless `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are `1`. Where a start is
  seeded, the k-means partial sums depend on the order threads finish.
* **Data.** Generated inside the script by `numpy.random.RandomState(10)`: a
  three-state Markov chain, two sequences of 300 and 200 samples, with Gaussian
  (three features and one), Gaussian mixture, categorical, multinomial and
  Poisson emissions. Two further Gaussian families carry a state that collapses
  onto repeated rows and a feature with no spread, so that a covariance floor
  binds. The hash of every input is recorded and compared.
* **Fits.** All seven model classes, every covariance type, both
  implementations (`"log"`, `"scaling"`), a fixed start (`init_params=""`,
  parameters set from the generator's values moved by a fixed amount) and a
  seeded start (`random_state=0`), ten iterations. For `GaussianHMM` the
  `params` `"stmc"` (with and without `tol=-inf`), `"stm"`, `"st"` and `"stc"`.
* **Compared.** Every fitted attribute, `monitor_.history`, the score, both
  decodes, `predict_proba`, a sample drawn with `random_state=1`, `covars_`, the
  parameters, and the exception type of a call that raises. The sufficient
  statistic of the last M-step is compared with `state_occupancy_`. Models
  pickled by 0.3.3 are loaded by this package and probed again. `normalize` and
  `fill_covars` are called on fixed arrays.
* **Exceptions.** A difference is accepted only if its exception is passed to
  `--allow` and it has exactly the shape the exception describes. Anything
  else is a difference.

| Id | Item | What is excused |
|---|---|---|
| E1 | spherical `covars_` | the shape of `_covars_` and `covars_` of a spherical `GaussianHMM`; every value equal to 0.3.3's |
| E2 | spherical `sample()` | rows drawn in a state other than the first, when there is more than one feature; the states and the first state's rows equal |
| E3 | MAP decode | the log-probability, equal to the log of the posteriors 0.3.3 summed, recomputed from `predict_proba` |
| E4 | covariances without means | `params` with `"c"` and not `"m"`: not compared, the fit must not raise and every array must be finite |
| E5 | `normalize` axis 0 | zero-sum slices: NaN before, 0 now |
| E6 | additions | `state_occupancy_` and the parameter `covars_floor` with value `None` |

### Running it

In two environments with the same versions, one with hmmlearn 0.3.3 built from
its tag and one with this package installed:

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python -I scripts/bit_equality.py dump --package hmmlearn --out REF
    python -I scripts/bit_equality.py dump --package hmmcraft --out FORK \
        --load-pickles REF
    python -I scripts/bit_equality.py compare REF FORK --allow E1,E2,E3,E4,E5,E6

The exit status is 0 for equal, 1 for a difference (each one is printed) and 2
when the dumps cannot be compared. The tests of the check run with
`python -m pytest scripts/test_bit_equality.py`.

### Result

Synthetic data as above (`RandomState(10)`, model seed 0, sample seed 1),
Python 3.12.3, Linux x86_64, one thread, 374 fits and the utility arrays, the
tree at version `0.3.3.post30+g0a5238ebe` (the last commit that changes the
package or the check; later commits change documentation, docstrings
included, or add the probe only), every exception listed above allowed:

| Environment | numpy | scipy | scikit-learn | Result |
|---|---|---|---|---|
| minimum versions | 1.26.0 | 1.11.2 | 1.3.1 | equal apart from E1 to E6 |
| newest versions | 2.5.4 | 1.18.1 | 1.9.1 | equal apart from E1 to E6 |

This is the whole claim: no result is stated for other data, other versions,
other thread counts or another platform.

With `covars_floor` left at `None`, the fits where a floor would bind
(`collapse` and `flat` families) are equal to 0.3.3's. With hmmlearn pull
request 588 applied alone, without the option, the check fails on exactly
those fits.

## A correction

The message of the merge `2ee7c48` says the side branch keeps the one commit
that changes fits off the first-parent history, "where every commit is
bit-equal to hmmlearn 0.3.3 apart from the named exceptions". That was wrong
for `e3e09b2`, which is on the first-parent history: two spherical
`fill_covars` cases differed from 0.3.3 by one ulp until `5754321` restored the
exact values. The check at the tree above, which includes `5754321`, shows no
such difference.
