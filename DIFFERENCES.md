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
| Invalid `lengths` | `lengths` must be a one-dimensional array of positive integers: an empty or two-dimensional array, a single integer, or an entry that is zero, negative, a float or a boolean raises `ValueError` in `fit` (before its start is computed), `score`, `score_samples`, `decode`, `predict` and `predict_proba`. 0.3.3 flattened a two-dimensional array and read a single integer as one sequence (the variational models' `fit` raised `TypeError` on a two-dimensional list and on a single integer), counted a boolean entry as 1 and raised `TypeError` for a float. It accepted zero and negative entries when the total matched. `numpy.split` reads a negative entry as an index from the end, so some split `X` differently; a zero entry, and some negative ones, gave the kernel a sequence without samples, and the kernel then indexed outside its arrays: `score` and `decode` returned wrong values with no error, `score` with `implementation="scaling"` raised an underflow error, and `fit` raised once it iterated. The kernel's functions now also raise `ValueError` for a matrix with no rows or no states. Valid `lengths` give the same results. | none |
| Invalid parameters and input | `ValueError` naming the parameter for: a `GaussianHMM` `means_` not of shape `(n_components, n_features)`; a `GMMHMM` `means_` that is not three-dimensional when `n_features` is not set; an `implementation` other than `"log"` or `"scaling"` (in `fit`, `score`, `score_samples`, `predict_proba` and `decode(algorithm="map")`); an `n_components` less than 1 (in `fit`, before its start). `sample()` of fewer than one sample also raises; a number of states or of samples of at least 1 that is not an integer, `n_components=True` included, fails as it did. 0.3.3 raised `IndexError`, `KeyError`, `ZeroDivisionError`, `TypeError` or a `ValueError` that did not name the parameter, scored a `GaussianHMM` whose `means_` had one row, and returned one sample for `sample(0)` and `sample(-1)`. Before `n_features` is known, `aic` and `bic` return their value, with the number of features taken from the emission parameter; 0.3.3 raised `AttributeError` (`TypeError` for a `CategoricalHMM`). The `covars_` of a `GaussianHMM` or a `VariationalGaussianHMM` returns its value then too, where 0.3.3 raised `AttributeError`: a `diag`, `full` or `tied` one needs no number of features, so `hasattr(model, "covars_")` is true where it was false; a `spherical` one takes it from `means_` (`means_posterior_` for the variational model), and with neither the means nor `n_features` set, or with means of the wrong shape, raises `ValueError` naming the parameter, from `hasattr(model, "covars_")` too, which returned false. Inside `fit` the number of features is set before `covars_` is read. A `MultinomialHMM` or `PoissonHMM` whose `n_features` is set to `None` takes it from the emission parameter; 0.3.3 raised `ValueError`. In the five expectation-maximization models, `fit`, `score`, `score_samples`, `predict`, `predict_proba`, `decode` and `sample` raise `ValueError` naming the parameter for a negative entry in `startprob_`, `transmat_`, `emissionprob_`, `weights_` or `lambdas_`, and for NaN or infinity in `means_`, `covars_`, `lambdas_` or a `MultinomialHMM` `emissionprob_`; 0.3.3 scored NaN and decoded states, or raised a `ValueError` that did not name the parameter. Zeros stay valid, and so does a sum or an entry above 1 by no more than the sum test's tolerance. A `full` or `tied` `GMMHMM` `covars_` holding NaN raises that it must contain only finite values, where 0.3.3 said it was not symmetric. The rows of a `MultinomialHMM` `emissionprob_` must sum to 1, as those of a `CategoricalHMM` must: 0.3.3 scored a row that does not, such as one rounded to three decimals (`[0.333, 0.333, 0.333]`), as NaN with scipy 1.18.1 and as a finite value with scipy 1.11.2, whose multinomial distribution replaces the last probability so that the row sums to 1; `fit` from such a row, `score` and `predict` now raise. The `GMMHMM` M-step does not clip the mixture weights at 0, so a fit with a `weights_prior` below 1 can return a negative entry in `weights_`; the next `score`, `decode`, `sample` or refit raises naming `weights_`, where 0.3.3 scored NaN and decoded states. A `spherical` or `diag` `GaussianHMM` whose starting covariance overflows (a sample of magnitude above about 1e154) raises naming `covars_` at the start of `fit`, as `full` and `tied` already raised, without naming it; 0.3.3 returned a model whose every parameter was NaN. Valid models give the same results. | none |
| `MultinomialHMM` | With no `emissionprob_` and `"e"` not in `init_params`, `fit` starts `emissionprob_` from `random_state`, as the other models start a parameter that is not set; 0.3.3 raised `AttributeError`. With `emissionprob_` set and `"e"` in `init_params`, `fit` logs that it is overwritten, as the other models do. The constructor, and so `clone`, no longer log the warning hmmlearn 0.2.8 added with the class. | none |
| Log-determinant | The `full` and `tied` log-densities take the log of the Cholesky factor's diagonal in place on a contiguous copy (`numpy.log(d, out=d)`), not of the strided diagonal. On a CPU with AVX-512, numpy before 2.0.2 has two float64 `log` loops, a vector loop and a scalar one. It gives a strided input the scalar loop when the output starts after the input's start but no more than one stride times the input's length past it, whether or not the two overlap, so hmmlearn 0.3.3 gets one loop or the other by where the allocator places the output. The two loops are different code, so their last bits can differ, and 0.3.3 could then refit one model to different last bits in one process; no run has reproduced such a refit. When the output is the input itself, both AVX-512 loops numpy before 2.0.2 dispatches to take the vector loop whatever the allocator, so this package always gets it there, and its last bits can differ from 0.3.3's. With AVX-512 and numpy 2.0.2 or later, the strided diagonal and the copy both get the vector loop; without AVX-512, both get the scalar one. The check below shows no change in either case, for the versions it names. The boundary and the loop choice were read in the float64 `log` loops of the Linux x86_64 wheels: numpy 1.26.0 and 2.0.1 measure a strided input one stride past its last element, numpy 2.0.2, 2.1.0 and 2.5.4 up to it. `scripts/probe_log_loops.py` shows whether a machine has the two loops. In both cells of `scripts/ci_bit_equality.sh`, on Linux x86_64 CPUs with AVX-512 (`avx512f`) and Python 3.12.16, it found that with numpy 1.26.0 some of its seeded values get a log from the strided input, output right after it, that differs as bytes from the log in place; none do once numpy's AVX-512 targets are turned off, and none with numpy 2.5.4 with them on. | none |
| Covariances without means | A `GaussianHMM` with `"c"` but not `"m"` in `params` accumulates the state mass it needs, which 0.3.3 did not. With the default `covars_weight=1` and `means_weight=0`, variances (`covars_prior` + the posterior-weighted sum of squares) / 1e-5, for `spherical` averaged over the features: a division by the 1e-5 floor with no posterior mass. For `full` and `tied`, infinite covariances after the first M-step and `ValueError` from the second iteration, when `covars_weight` is at most `n_features`. `scripts/probe_covariance_only_fit.py` checks each clause on hmmlearn 0.3.3 built from its tag (Python 3.12.3, one thread; numpy 1.26.0, scipy 1.11.2, scikit-learn 1.3.1, threadpoolctl 3.8.0, and numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, threadpoolctl 3.8.0). This is one of the two changes to a fit that have no option; the other is the log-determinant above. | none |
| `state_occupancy_` | New attribute after `fit`, for every model: the expected number of samples in each state, from the last E-step. | none (new) |
| `covars_floor` | New `GaussianHMM` parameter. A positive number floors the covariances after every M-step that updates them: variances (`"spherical"`, `"diag"`) and eigenvalues (`"full"`, `"tied"`) below it are raised to it. It is independent of `min_covar`, which keeps its one effect, the term added to the covariance that starts the fit. | `covars_floor=None` (default): 0.3.3's fit |

A model pickled by hmmlearn 0.3.3 loads with `covars_floor=None`, and for a
spherical model stores the same variances as before.

A spherical `GaussianHMM` whose `covars_` is set to an array of shape
`(n_components, n_features)` with unequal values in a row keeps those values:
`score` and `decode` use each feature's own variance, as a `diag` model would,
while `covars_` reports each state's mean variance. hmmlearn 0.3.3 scores such
a model the same way and reports one matrix per state and feature.

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
  otherwise. The floor cell of `scripts/ci_bit_equality.sh` sets numpy's
  `NPY_DISABLE_CPU_FEATURES` to turn its AVX-512 loops off, and stops if numpy
  still reports one on: with them, numpy before 2.0.2 leaves hmmlearn 0.3.3's
  `full` and `tied` log-densities to memory placement (the log-determinant
  above), and the two packages would not take the same loop.
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
tree at version `0.3.3.post47+g489eb03db` (the last commit that changes the
package or the check; later commits change documentation only), every
exception listed above allowed:

| Environment | numpy | scipy | scikit-learn | Result |
|---|---|---|---|---|
| minimum versions | 1.26.0 | 1.11.2 | 1.3.1 | equal apart from E1 to E6 |
| newest versions | 2.5.4 | 1.18.1 | 1.9.1 | equal apart from E1 to E6 |

With the log-determinant row above (the log in place on a contiguous copy)
added to that tree's package, the check gives the same result in both
environments, on the same data, versions and thread count, on a Linux x86_64
CPU without AVX-512.

The package with that change, run by `scripts/ci_bit_equality.sh` on
GitHub-hosted Linux x86_64 runners whose CPU reports AVX-512 (`avx512f`), AMD
and Intel, gives the same result on the same data and thread count, with
Python 3.12.16, threadpoolctl 3.8.0 and the versions in the table: in the
minimum-versions cell with numpy's AVX-512 targets turned off, as the script
does, and in the newest-versions cell with them on.

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
