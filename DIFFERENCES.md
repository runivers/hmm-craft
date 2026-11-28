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
| Invalid parameters and input | `ValueError` naming the parameter for: a `GaussianHMM` `means_` not of shape `(n_components, n_features)`; a `GMMHMM` `means_` that is not three-dimensional when `n_features` is not set; an `implementation` other than `"log"` or `"scaling"` (in `fit`, `score`, `score_samples`, `predict_proba` and `decode(algorithm="map")`); an `n_components` less than 1 (in `fit`, before its start). `sample()` of fewer than one sample also raises; a number of states or of samples of at least 1 that is not an integer, `n_components=True` included, fails as it did. 0.3.3 raised `IndexError`, `KeyError`, `ZeroDivisionError`, `TypeError` or a `ValueError` that did not name the parameter, scored a `GaussianHMM` whose `means_` had one row, and returned one sample for `sample(0)` and `sample(-1)`. Before `n_features` is known, `aic` and `bic` return their value, with the number of features taken from the emission parameter; 0.3.3 raised `AttributeError` (`TypeError` for a `CategoricalHMM`). The `covars_` of a `GaussianHMM` or a `VariationalGaussianHMM` returns its value then too, where 0.3.3 raised `AttributeError`: a `diag`, `full` or `tied` one needs no number of features, so `hasattr(model, "covars_")` is true where it was false; a `spherical` one takes it from `means_` (`means_posterior_` for the variational model), and with neither the means nor `n_features` set, or with means of the wrong shape, raises `ValueError` naming the parameter, from `hasattr(model, "covars_")` too, which returned false. Inside `fit` the number of features is set before `covars_` is read. A `MultinomialHMM`, `PoissonHMM` or `GMMHMM` whose `n_features` is set to `None` takes it from the emission parameter; 0.3.3 raised `ValueError` (`TypeError` for `GMMHMM`). `fit` reads such an `n_features` as not set, for `GaussianHMM` and `VariationalGaussianHMM` too, where 0.3.3 raised "Unexpected number of dimensions, got 3 but expected None". A `VariationalGaussianHMM` without `n_features` takes the number of features from `means_posterior_`, where 0.3.3 raised `AttributeError`. When `fit` raises before its first iteration, `n_features` is as it was before the call unless the emission parameter gives the number the call set (`GaussianHMM` keeps the width of its `means_`, as before), and the priors of a `GMMHMM` are as they were. A `covars_prior` or `covars_weight` that the call filled in from `None` keeps the number it was filled with, without a shape. The parameters the start drew, and the `n_trials` of a `MultinomialHMM`, stay as the call left them. 0.3.3 kept the number of features of the refused `X` and the priors broadcast to it, so the next `fit` of a `MultinomialHMM`, `PoissonHMM`, `GMMHMM` or `VariationalGaussianHMM` on `X` of the model's width raised "Unexpected number of dimensions". Four consequences: after such a refusal, `n_features` reads as before the call, and so does what derives from it, such as the shape of a `spherical` `VariationalGaussianHMM`'s `covars_`, where 0.3.3 gave the refused width; `del model.n_features` raises `AttributeError`, while `model.n_features = 3` works as it did; `get_params()` and `clone()` show a filled covariance prior as a number where 0.3.3 showed an array, and fits are the same; a `full` or `tied` `GMMHMM` refitted on `X` of another width fits with the `covars_weight` filled for the refused width, where 0.3.3 raised, not with a new model's default. In the five expectation-maximization models, `fit`, `score`, `score_samples`, `predict`, `predict_proba`, `decode` and `sample` raise `ValueError` naming the parameter for a negative entry in `startprob_`, `transmat_`, `emissionprob_`, `weights_` or `lambdas_`, and for NaN or infinity in `means_`, `covars_`, `lambdas_` or a `MultinomialHMM` `emissionprob_`. Apart from the `MultinomialHMM` rows described below, 0.3.3 scored NaN and decoded states, or raised a `ValueError` that did not name the parameter, except for two kinds of model that it could score as a finite number and that are now refused. With `implementation="scaling"`, a negative entry in `startprob_`, `transmat_` or a `CategoricalHMM` `emissionprob_` could give a finite number, computed with the negative probability. An infinite entry in the `means_` or `covars_` of a `spherical` or `diag` `GaussianHMM` or `GMMHMM` gave that state, or that mixture component, probability zero for every sample: `score`, `score_samples`, `predict`, `predict_proba` and `decode` returned finite values when the other states or components could emit every sample, and otherwise a score of minus infinity with NaN state probabilities (the underflow `ValueError` with `"scaling"`); a `full` or `tied` model raised SciPy's "array must not contain infs or NaNs". An infinite `lambdas_` entry did the same when every count of that feature in `X` was zero, and scored NaN otherwise. In a `spherical` or `diag` model, a `fit` started from an infinite mean or variance returned NaN parameters with the default `params` and priors, and with others a `transmat_` row of zeros that the next call refused, or a finite fit; `sample` drew infinite observations from an infinite mean, and from an infinite variance of a `diag` model raised `LinAlgError` with numpy 2.5.4 and drew NaN with numpy 1.26.0. A finite entry is not refused, however large. In `VariationalCategoricalHMM` and `VariationalGaussianHMM`, `score`, `score_samples`, `predict`, `predict_proba`, `decode` and `sample` raise `ValueError` naming the parameter for a negative entry, NaN or infinity in `startprob_`, `transmat_` or a `VariationalCategoricalHMM` `emissionprob_`, and for NaN in a `VariationalGaussianHMM` `means_posterior_` or `covars_`; `fit` does so for NaN or infinity in a prior or posterior, `means_posterior_` included (an infinite `beta_posterior_` only when `"m"` is not in `params`). `fit` also raises it for a `startprob_prior_` or `startprob_posterior_` not of shape `(n_components,)` and a `transmat_prior_` or `transmat_posterior_` not of shape `(n_components, n_components)`; with one state, a `VariationalGaussianHMM` `startprob_prior_` given as a number and a `transmat_prior_` of shape `(1,)` stay accepted, as they fit as those shapes do. 0.3.3 scored NaN or infinity and decoded states, or raised a NumPy or SciPy error that did not name the parameter, and its `fit` returned NaN parameters or a lower bound that is not finite, or raised a NumPy error. 0.3.3's `VariationalGaussianHMM` did not test those shapes: most raised an error that did not name the array, and with more than one state four fitted with the arrays of a documented prior and lower bounds, which the convergence test reads, of another one: a `startprob_prior_` of shape `(1,)` or given as a number, which the M-step broadcasts and the bound sums as stored, and a `transmat_prior_` of shape `(n_components,)`, whose entry `j` the M-step adds to column `j` and the bound reads as the prior of row `j`, or of shape `(n_components, 1)`, whose entry `i` the M-step adds to each entry of row `i` and the bound reads as a prior of one entry. Its `VariationalCategoricalHMM` tested the length of the start arrays and the shape of the transition ones, in `fit` and on prediction, as it still does: a start array of shape `(n_components, 1)` or `(n_components, n_components)` passed, and `fit` raised an error that did not name the array. A `VariationalCategoricalHMM` `startprob_prior_` or `startprob_posterior_` given as a number raised `TypeError`, in `fit` and on prediction; it now raises the length `ValueError` in both, and no returned value moves. What these models do not refuse: a prediction tests the value of no prior or posterior but `means_posterior_`, and reads no other but their shapes and, for a `VariationalCategoricalHMM` without `n_features`, the width of `emissionprob_posterior_`; `fit` does not refuse a negative prior or posterior, nor an infinite `beta_posterior_` when `"m"` is in `params`, as its first M-step replaces it; a prediction does not refuse rows of `startprob_`, `transmat_` or `emissionprob_` that do not sum to 1, nor an infinite `means_posterior_` or `covars_` (in a `spherical` or `diag` model that state then emits nothing and the score is finite), and gives 0.3.3's results for them; a prediction uses none of the start and transition priors and posteriors: a `VariationalGaussianHMM` one does not test their shapes, so it returns 0.3.3's results with a shape `fit` refuses, and a `VariationalCategoricalHMM` one returns them for the start arrays its length test passes. Zeros stay valid, and so does a sum or an entry above 1 by no more than the sum test's tolerance. `GaussianHMM.fit` raises it too, naming the prior, at the start of the fit whatever `n_iter` is, for a `means_prior`, `means_weight`, `covars_prior` or `covars_weight` of a form the constructor does not document, among those `params` lets the M-step read (`means_prior` and `means_weight` with `"m"` or `"c"`, the others with `"c"`). A prior must broadcast, as `np.broadcast_to` does, without an added axis, to `(n_components, n_features)`; a `full` `covars_prior` to `(n_components, n_features, n_features)` and a `tied` one to `(n_features, n_features)`, symmetric as `np.allclose` tests it. `covars_weight`, and `means_weight` with `full` or `tied` and `"c"` in `params`, must be a number, which an array with one element and at most two dimensions is (three for a `full` or `tied` `covars_weight`); an array of another shape is refused even when its entries are all equal. A `tied` `covars_prior` that broadcasts to `(1, n_features, n_features)` is accepted too. 0.3.3 fitted some of these to a model its documents rule out: one `tied` covariance per state; a `full` or `tied` covariance that is not symmetric, or whose `means_weight` weighs each entry of the outer product apart; means and variances with `n_components` columns in a one-feature model; `diag` variances with an added axis, which `covars_` and `sample()` then read wrongly. Others, such as an array of equal entries where a number is documented, fitted as the number. The rest raised a NumPy `ValueError` that did not name the prior. When the model keeps its `means_` and `X` has another width, a prior of the width of `X` is now named before `X` is. A prediction does not read the priors and is unchanged. A `full` or `tied` `GMMHMM` `covars_` holding NaN raises that it must contain only finite values, where 0.3.3 said it was not symmetric. The rows of a `MultinomialHMM` `emissionprob_` must sum to 1, as those of a `CategoricalHMM` must: 0.3.3 scored a row that does not, such as one rounded to three decimals (`[0.333, 0.333, 0.333]`), as NaN with scipy 1.18.1 and as a finite value with scipy 1.11.2, whose multinomial distribution replaces the last probability so that the row sums to 1; `fit` from such a row, `score` and `predict` now raise. The `GMMHMM` M-step does not clip the mixture weights at 0, so a fit with a `weights_prior` below 1 can return a negative entry in `weights_`; the next `score`, `decode`, `sample` or refit raises naming `weights_`, where 0.3.3 scored NaN and decoded states. A `spherical` or `diag` `GaussianHMM` whose starting covariance overflows (a sample of magnitude above about 1e154) raises naming `covars_` at the start of `fit`, as `full` and `tied` already raised, without naming it; 0.3.3 returned a model whose every parameter was NaN. `score`, `score_samples`, `predict`, `predict_proba`, `decode`, `aic` and `bic` raise `ValueError` naming `X` when its number of columns is not the model's number of features (not tested for `CategoricalHMM` and `VariationalCategoricalHMM`, whose `X` holds symbols), when a symbol of those two models is negative, not a whole number or not below `n_features`, and when a `MultinomialHMM` or `PoissonHMM` count is negative or not a whole number. `fit` does so for `PoissonHMM` counts and for a `GaussianHMM` that keeps its `means_` (below), and otherwise raises the errors it raised before. 0.3.3 broadcast an `X` of one column against every feature of a `GaussianHMM`, `GMMHMM` or `VariationalGaussianHMM` of every covariance type and of a `PoissonHMM`, and an `X` of any width against a one-feature `spherical` or `diag` model of those three classes or a one-feature `PoissonHMM`, and returned scores and states; it raised a NumPy or SciPy error that did not name `X` for a one-feature `full` or `tied` model and for the other widths. It read a negative symbol from the end of `emissionprob_` (`-1` scored as the last symbol), raised `IndexError` for a symbol of `n_features` or more and for a fractional one, and SciPy's `ValueError` for a `MultinomialHMM` `X` of another width. It scored a negative or fractional count as impossible, `-inf` with `implementation="log"` and an underflow `ValueError` with `"scaling"`, and fitted a `PoissonHMM` to such counts with every rate NaN (`"log"`; the underflow error with `"scaling"`). With `"l"` in `init_params`, an `X` whose mean is negative still raises NumPy's `scale < 0` from the start of `fit`, before `X` is checked. `fit` of a `GaussianHMM` that keeps its `means_` (`"m"` not in `init_params`) refuses `X` of another width the same way, after its start: given one column against two-feature `means_`, 0.3.3 returned a fitted `diag`, `spherical` or `tied` model (a `diag` one with equal columns in `means_`) and raised SciPy's "shapes ... incompatible" for `full`; a wider `X` raised NumPy's broadcast error. `CategoricalHMM` and `VariationalCategoricalHMM` refuse a boolean `X` on prediction, as `fit` refuses it: a refusal on the dtype, not the values, because NumPy reads a boolean array as a mask over the symbols, not as the symbols 0 and 1. With as many entries as `n_features`, 0.3.3 returned one state per `True` entry and a score that is not the score of the symbols; other lengths raised `IndexError`. Boolean counts stay valid for `MultinomialHMM` on prediction (its `fit` refuses them, as it did) and for `PoissonHMM` in `fit` and on prediction. Whole numbers stored as floats stay valid counts; a `MultinomialHMM` row whose counts do not add up to `n_trials` is still scored with its own total; a `CategoricalHMM` given whole numbers stored as floats raises `IndexError` on prediction, as it did. Valid models and valid `X` give the same results. | none |
| `MultinomialHMM` | With no `emissionprob_` and `"e"` not in `init_params`, `fit` starts `emissionprob_` from `random_state`, as the other models start a parameter that is not set; 0.3.3 raised `AttributeError`. With `emissionprob_` set and `"e"` in `init_params`, `fit` logs that it is overwritten, as the other models do. The constructor, and so `clone`, no longer log the warning hmmlearn 0.2.8 added with the class. | none |
| What a prediction leaves in the model | In every model, `score`, `score_samples`, `predict`, `predict_proba`, `decode` and `sample`, and `aic` and `bic` where the model has them, check the model and write nothing to it. 0.3.3's check stored `startprob_`, `transmat_` and the emission parameter again, converted to arrays (for `GMMHMM`, a copy of `weights_`, `means_` and `covars_` on every call), set `n_features`, and for `GMMHMM` filled in the defaults of `covars_prior` and `covars_weight` and stored every prior broadcast to an array. 0.3.3's check of the variational models stored their priors and posteriors again, converted to arrays (float arrays for the emission ones of `VariationalGaussianHMM`, whose degrees of freedom stay a number when `tied`; that model's start and transition priors and posteriors were never converted), and set the missing `n_features` of a `VariationalCategoricalHMM` to the width of `emissionprob_posterior_`. A parameter is now converted when it is assigned, by the function the check used, where 0.3.3 converted it at the first check (at every check for the variational models): after `model.startprob_ = p` with `p = [0.6, 0.4]`, a later change to `p` is not read, where 0.3.3 read it until the first prediction (the next `fit` or prediction for the variational models, and always for the start and transition priors and posteriors of a `VariationalGaussianHMM`, which it never converted); the same holds for an array given to the `weights_`, `means_` or `covars_` of a `GMMHMM`, which is copied, and for an array other than float64, of integers for example, given to the means, beta, scale or degrees-of-freedom prior or posterior of a `VariationalGaussianHMM`, which the conversion to floats copies. The variational models also convert `startprob_`, `transmat_` and `emissionprob_`, which 0.3.3 never converted: a `VariationalCategoricalHMM` given `emissionprob_` as a list predicts, where 0.3.3 raised `TypeError`. A `VariationalGaussianHMM` given a start or transition posterior as a list, with `init_params` keeping it, fits; 0.3.3 raised `AttributeError`. A list that does not make an array, such as a ragged one, raises `ValueError` when it is assigned, or when a pickle holding it is loaded, not at the first prediction. A model pickled with a parameter as a list, as 0.3.3 pickles a model built by hand and never checked, loads with it as an array. After a prediction on a model built by hand, `n_features` is not set (it stays `None` for a `CategoricalHMM` or `VariationalCategoricalHMM` built without it), and `get_params()` shows the `GMMHMM` priors as they were given; a prior that does not broadcast to its shape is still refused on prediction. A `fit` after a prediction is the `fit` without it. A fitted `GaussianHMM` given `means_` and `covars_` of another width raises "Unexpected number of dimensions" in `fit` on `X` of that width; 0.3.3 fitted it after a prediction, which had set `n_features`. A `CategoricalHMM` or `VariationalCategoricalHMM` built without `n_features`, with `"e"` not in `init_params`, takes the number of symbols in `fit` from its `emissionprob_` (`emissionprob_posterior_`), with or without a `score` before it, as 0.3.3 did after a prediction. An `X` that shows fewer symbols fits, where 0.3.3 raised without a prediction that `emissionprob_` must have the shape of the symbols seen; a larger symbol raises, naming it. With `"e"` in `init_params` that `CategoricalHMM` starts as many symbols as `X` shows, where after a prediction 0.3.3 kept the number of symbols of the `emissionprob_` it replaced. A `VariationalCategoricalHMM` built without `n_features`, with `"e"` in `init_params`, also starts as many symbols as `X` shows after a `score`, where 0.3.3 started as many as the `emissionprob_posterior_` it replaced. `get_stationary_distribution()` returns its value for a `transmat_` given as a list before any prediction; 0.3.3 raised `AttributeError` until one. Once a `means_` of a valid shape is set on a `GaussianHMM`, its width gives the number of features to `aic`, `bic` and a `spherical` `covars_`, before any prediction too; 0.3.3 used an `n_features` left from an earlier fit until the first prediction replaced it, so its first `aic` could differ from the next. `sample()` without `random_state` still advances the model's own random state. Predictions give the same results. | none |
| Log-determinant | The `full` and `tied` log-densities take the log of the Cholesky factor's diagonal in place on a contiguous copy (`numpy.log(d, out=d)`), not of the strided diagonal. On a CPU with AVX-512, numpy before 2.0.2 has two float64 `log` loops, a vector loop and a scalar one. It gives a strided input the scalar loop when the output starts after the input's start but no more than one stride times the input's length past it, whether or not the two overlap, so hmmlearn 0.3.3 gets one loop or the other by where the allocator places the output. The two loops are different code, so their last bits can differ, and 0.3.3 could then refit one model to different last bits in one process; no run has reproduced such a refit. When the output is the input itself, both AVX-512 loops numpy before 2.0.2 dispatches to take the vector loop whatever the allocator, so this package always gets it there, and its last bits can differ from 0.3.3's. With AVX-512 and numpy 2.0.2 or later, the strided diagonal and the copy both get the vector loop; without AVX-512, both get the scalar one. The check below shows no change in either case, for the versions it names. The boundary and the loop choice were read in the float64 `log` loops of the Linux x86_64 wheels: numpy 1.26.0 and 2.0.1 measure a strided input one stride past its last element, numpy 2.0.2, 2.1.0 and 2.5.4 up to it. `scripts/probe_log_loops.py` shows whether a machine has the two loops. In both cells of `scripts/ci_bit_equality.sh`, on Linux x86_64 CPUs with AVX-512 (`avx512f`) and Python 3.12.16, it found that with numpy 1.26.0 some of its seeded values get a log from the strided input, output right after it, that differs as bytes from the log in place; none do once numpy's AVX-512 targets are turned off, and none with numpy 2.5.4 with them on. | none |
| Covariances without means | A `GaussianHMM` with `"c"` but not `"m"` in `params` accumulates the state mass it needs, which 0.3.3 did not. With the default `covars_weight=1` and `means_weight=0`, variances (`covars_prior` + the posterior-weighted sum of squares) / 1e-5, for `spherical` averaged over the features: a division by the 1e-5 floor with no posterior mass. For `full` and `tied`, infinite covariances after the first M-step and `ValueError` from the second iteration, when `covars_weight` is at most `n_features`. `scripts/probe_covariance_only_fit.py` checks each clause on hmmlearn 0.3.3 built from its tag (Python 3.12.3, one thread; numpy 1.26.0, scipy 1.11.2, scikit-learn 1.3.1, threadpoolctl 3.8.0, and numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, threadpoolctl 3.8.0). This is one of the three changes to a fit that have no option; the others are the log-determinant above and the k-means start below. | none |
| k-means start | Where a start comes from k-means (the `means_` of a `GaussianHMM` with `"m"` in `init_params` or no `means_`; the `means_` of a `GMMHMM`; for a `VariationalGaussianHMM`, `means_posterior_` and `beta_posterior_` under `"m"` and `dof_posterior_` and `scale_posterior_` under `"c"`), scikit-learn's `KMeans` runs inside `threadpoolctl.threadpool_limits(limits=1)`. Its Lloyd iterations add per-thread partial sums, so with two or more threads 0.3.3's start can depend on the number of threads, and from three threads on, on the order in which they finish. With a fixed `random_state`, this package's start is the one-thread start at every thread count; at one thread it is the start 0.3.3 computes, and the check below compares the two. For a `GMMHMM`, one limit holds every k-means call of the start and the steps between them: where a cluster has fewer samples than `n_mix`, the covariance of `X` (plus `min_covar` on the diagonal) that the cluster's means are drawn with is computed under the limit too, and under `"c"`, `covars_` starts from that covariance. The limit covers the thread pools threadpoolctl controls (OpenBLAS, MKL, BLIS, FlexiBLAS and the OpenMP runtimes); a library outside that list keeps its own threads. The limits in force before are restored once the start's k-means calls are done, when one fit runs at a time; OpenBLAS's limit applies to the whole process, so fits run at once from several Python threads can leave it changed, or leave part of a start unlimited. `src/hmmcraft/tests/test_kmeans_start.py` compares the start of the three classes, every covariance type, at two and four threads with one, on its synthetic data (`RandomState(0)`, 1200 samples, two features, `random_state=0`), with the thread variables at 1 and unset (Python 3.12.3, Linux x86_64; numpy 1.26.0, scipy 1.11.2, scikit-learn 1.3.1, threadpoolctl 3.8.0 and 3.5.0, and numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, threadpoolctl 3.8.0 and 3.5.0). | none |
| Underflow with `implementation="scaling"` | A sequence whose scaled forward pass underflows, because the probabilities of one of its samples sum to less than 1e-300 over the states, is computed with the log functions: in `score`, `score_samples`, `predict_proba`, `aic`, `bic`, `decode` and `predict` with `algorithm="map"`, and in each iteration of `fit`, for every model class. 0.3.3 raised `ValueError` ("forward pass failed with underflow"). The result for that sequence, and in `fit` its statistics, are those of `implementation="log"`; the other sequences are computed as before. One warning is logged (logger `hmmcraft.base`) per call and per iteration of `fit` in which a sequence was computed this way; `decode(algorithm="map")` logs one per such sequence. When the log functions give no finite log-likelihood either, as for a sample with probability zero in every state, the same `ValueError` is raised. A subclass that overrides `_accumulate_sufficient_statistics` receives the log-space `lattice`, `fwdlattice` and `bwdlattice` for such a sequence while `implementation` is `"scaling"`. Calls that returned give the same results. | none |
| `state_occupancy_` | New attribute after `fit`, for every model: the expected number of samples in each state, from the last E-step. | none (new) |
| `covars_floor` | New `GaussianHMM` parameter. A positive number floors the covariances after every M-step that updates them: variances (`"spherical"`, `"diag"`) and eigenvalues (`"full"`, `"tied"`) below it are raised to it. It is independent of `min_covar`, which keeps its one effect, the term added to the covariance that starts the fit. | `covars_floor=None` (default): 0.3.3's fit |
| `n_threads` | New parameter of every model: `fit` computes the forward and backward passes and the expected transition counts of the sequences of a block on that many threads, each sequence on one thread as it is with one; the statistics are added in `lengths` order. Above one thread a block holds at least that many sequences. Prediction uses one thread. | `n_threads=1` (default): no thread is started; 2 and 4 give the fits of 1 in the check below |

A model pickled by hmmlearn 0.3.3 loads with `covars_floor=None` and
`n_threads=1`, and for a spherical model stores the same variances as before.

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
  `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are `1`, and fits every case
  inside `threadpool_limits(limits=1)`. Where a start is seeded, hmmlearn
  0.3.3's k-means partial sums depend on the order threads finish, and this
  package's start is its one-thread start (the k-means row above), so the two
  are compared at one thread. The comparison leaves `n_threads` at 1; `dump
  --n-threads N` fits this package's models with `n_threads=N`, a dump
  compared with the one-thread dump of the same environment, file by file.
* **Data.** Generated inside the script by `numpy.random.RandomState(10)`: a
  three-state Markov chain, two sequences of 300 and 200 samples, with Gaussian
  (three features and one), Gaussian mixture, categorical, multinomial and
  Poisson emissions. Two further Gaussian families carry a state that collapses
  onto repeated rows and a feature with no spread, so that a covariance floor
  binds. Three more copy the three-feature Gaussian, the Gaussian mixture and
  the Poisson families with one sample of the first sequence (row 50) set to
  1000, far from every state. A further family draws the three-state chain
  and the three-feature Gaussian, Gaussian mixture, categorical, multinomial
  and Poisson emissions with `RandomState(10)` on 220 sequences whose lengths
  run through 1, 2, 3, 5, 8, 13, 21, 34, 55 and 89, 22 times, 5082 samples in
  all. Two more families are drawn with `RandomState(10)`, each from its own
  generator: a left-to-right chain, which starts in the first state and from
  each state stays or moves to the next with probability 0.5 until the last,
  which it keeps (the matrices of hmmlearn's tutorial), with the
  three-feature Gaussian and the categorical emissions, on 10 sequences of 10
  samples; and a four-feature Gaussian of the three-state chain on 40
  sequences whose lengths run through 1, 2, 3, 5, 8, 13, 21, 34, 55 and 89,
  four times, 924 samples in all. The hash of every input is recorded and
  compared.
* **Fits.** All seven model classes, every covariance type, both
  implementations (`"log"`, `"scaling"`), a fixed start (`init_params=""`,
  parameters set from the generator's values moved by a fixed amount) and a
  seeded start (`random_state=0`), ten iterations. For `GaussianHMM` the
  `params` `"stmc"` (with and without `tol=-inf`), `"stm"`, `"st"` and `"stc"`.
  On the short sequences every class is fitted the same way, `GaussianHMM`
  with `"stmc"` and `tol=-inf` only. Further cases fit `GaussianHMM`,
  `GMMHMM`, `PoissonHMM`, `CategoricalHMM` and `MultinomialHMM` from the fixed
  start with priors of their documented shapes, valued as in hmmlearn's
  `test_fit_with_priors` (`10 * p + 2` for a Dirichlet prior on probabilities
  `p`, the generating means and covariances with weight 2, plus
  `n_features` for the `full` and `tied` covariances; for `GMMHMM` the
  same for the mixture weights, so that none can become negative, and
  `covars_prior=1.0` as in hmmlearn's `test_gmm_hmm`; 2 for both `PoissonHMM`
  rate parameters); fit `GaussianHMM` and `CategoricalHMM` on the
  left-to-right families from its own start and transition matrix, zeros
  included, with those priors, and from the fixed start with priors below one
  (for `GaussianHMM` `1 / n_components` on the start and transitions, for
  `CategoricalHMM` `1 / n_features` on the emissions only, the variational
  models' defaults), so that the update's clip at zero decides some entries;
  fit `GaussianHMM` from the tutorial's start and transition matrix with its
  `init_params="mc"`; fit `GaussianHMM` and `GMMHMM` `full`
  and `tied` on the four-feature family as on the short sequences; and build
  the prior cases and the variational models with `set_params` after
  construction, `n_iter`, `tol` and `verbose` given to the constructor.
* **Underflow cases.** 33 cases with `implementation="scaling"` in which the
  scaled forward pass underflows: each predicts on the first sequence alone,
  with `"log"` as well, or fits on both sequences. The models start from the
  fixed start on the far families (`GaussianHMM`, `GMMHMM`, `PoissonHMM`),
  from the fixed start with the first emission probability of every state
  set to 1e-320 (`CategoricalHMM`, `MultinomialHMM`), or from the seeded
  scaling fit on the clean family, refitted without initialization (the
  variational models; for `VariationalCategoricalHMM` the first posterior
  emission count of every state set to 1e-5, and no prediction case). In two
  `CategoricalHMM` cases that probability is zero: both packages must raise
  the same error.
* **Compared.** Every fitted attribute, `monitor_.history`, the score, `aic`
  and `bic` where the model has them, both decodes, `predict_proba`, a sample
  drawn with `random_state=1`, `covars_`, the parameters, and the exception
  type of a call that raises. The sufficient
  statistic of the last M-step is compared with `state_occupancy_`. Each fit
  is repeated with a subclass that records the statistics; the package
  computes that fit sequence by sequence and the first one by blocks of
  sequences, and every array of the two must be equal. Models pickled by 0.3.3
  are loaded by this package and probed again. `normalize` and `fill_covars`
  are called on fixed arrays.
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
| E6 | additions | `state_occupancy_`, the parameter `covars_floor` with value `None` and the parameter `n_threads` with value 1 |
| E7 | scaling underflow | the underflow cases: 0.3.3 raises the underflow error and this package returns; a prediction on one sequence is finite and equal byte for byte to 0.3.3's result with `implementation="log"` (the MAP log-probability as E3 recomputes it); a fit must not raise and every array must be finite |

### Running it

In two environments with the same versions, one with hmmlearn 0.3.3 built from
its tag and one with this package installed:

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python -I scripts/bit_equality.py dump --package hmmlearn --out REF
    python -I scripts/bit_equality.py dump --package hmmcraft --out FORK \
        --load-pickles REF
    python -I scripts/bit_equality.py compare REF FORK \
        --allow E1,E2,E3,E4,E5,E6,E7

The exit status is 0 for equal, 1 for a difference (each one is printed) and 2
when the dumps cannot be compared. The tests of the check run with
`python -m pytest scripts/test_bit_equality.py`.

### Result

Synthetic data as above (`RandomState(10)`, model seed 0, sample seed 1),
Python 3.12.3, Linux x86_64, one thread, 374 fits and the utility arrays, the
tree at version `0.3.3.post47+g489eb03db` (the last commit that changes the
package or the check; later commits change documentation only), E1 to E6
allowed:

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

At the tree that adds the underflow cases and E7, which includes every change
to the package since the tree above, the check gives equal apart from E1 to E7
on the same data, with Python 3.12.3, one thread, 374 fits, the 33 underflow
cases and the utility arrays, in both environments: numpy 1.26.0, scipy
1.11.2, scikit-learn 1.3.1, and numpy 2.5.5, scipy 1.18.1, scikit-learn 1.9.1.

At the tree that computes the `diag` and `spherical` log-density and the
`full` and `tied` statistic in C++ and makes `GMMHMM` compute its mixture
densities once per E-step, which includes every change to the package since
the tree of the underflow cases, the check gives equal apart from E1 to E7 on
the same data, cases and versions as that tree, with Python 3.12.3, Linux
x86_64 and one thread.

At the tree that runs the k-means start on one thread (the k-means row above),
which includes every change to the package since the tree of the mixture
densities, the check gives equal apart from E1 to E7 on the same data and
cases, with Python 3.12.3, Linux x86_64, one thread and threadpoolctl 3.8.0,
in both environments: numpy 1.26.0, scipy 1.11.2, scikit-learn 1.3.1, and
numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1. With threadpoolctl 3.5.0 on
both sides in the first environment, the result is the same, and every array
of the package's dump is byte for byte the one with 3.8.0.

At the tree that adds the family of many short sequences, which changes the
check and its tests only, the check gives equal apart from E1 to E7 on the data
above, with Python 3.12.3, Linux x86_64, one thread, 428 fits (54 on the short
sequences), the 33 underflow cases and the utility arrays, in both
environments: numpy 1.26.0, scipy 1.11.2, scikit-learn 1.3.1, threadpoolctl
3.5.0, and numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, threadpoolctl 3.8.0.
In each environment, the package's results for the cases of the tree before
are byte for byte that tree's.

At the tree that computes the E-step by blocks of sequences, which includes
every change to the package since the tree of the short sequences, the check
gives equal apart from E1 to E7 on the same data and cases, with Python
3.12.3, Linux x86_64 and one thread, in both environments: numpy 1.26.0,
scipy 1.11.2, scikit-learn 1.3.1, threadpoolctl 3.5.0, and numpy 2.5.4,
scipy 1.18.1, scikit-learn 1.9.1, threadpoolctl 3.8.0. In each environment,
the package's results are byte for byte those of the tree before.

At the tree that factors each `full` and `tied` covariance of a `GaussianHMM`
once per E-step, which includes every change to the package since the tree
of the blocks, the check gives equal apart from E1 to E7 on the same data and
cases, with Python 3.12.3, Linux x86_64 and one thread, in both
environments: numpy 1.26.0, scipy 1.11.2, scikit-learn 1.3.1, threadpoolctl
3.5.0, and numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, threadpoolctl
3.8.0. In each environment, the package's results are byte for byte those of
the tree before.

At the tree that adds `n_threads`, which includes every change to the package
since the tree of the factored covariances, the check gives equal apart from
E1 to E7 on the same data and cases, with Python 3.12.3, Linux x86_64 and one
thread, in both environments: numpy 1.26.0, scipy 1.11.2, scikit-learn 1.3.1,
threadpoolctl 3.5.0, and numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1,
threadpoolctl 3.8.0. In each environment, the package's results are byte for
byte those of the tree before, and its dumps with `--n-threads 2` and
`--n-threads 4` (the thread variables still at 1) hold the bytes of its
one-thread dump. In the same environments,
`src/hmmcraft/tests/test_kernel_threads.py` compares, as bytes, the fits of
every model class on 2 and 4 threads and on more threads than sequences with
the fit on one thread, on its seeded data (`RandomState(10)`, the sequence
lengths of `src/hmmcraft/tests/test_estep_blocks.py`), and the block
functions' results on several threads with those of the per-sequence
functions (`RandomState(0)`).

At the tree that adds the prior, left-to-right, four-feature and `set_params`
cases and the `aic` and `bic` probes, which changes the check and its tests
only, the check gives equal apart from E1 to E7 on the data above, with Python
3.12.3, Linux x86_64, one thread, 526 fits (98 in the added families), the 33
underflow cases and the utility arrays, in both environments: numpy 1.26.0,
scipy 1.11.2, scikit-learn 1.3.1, threadpoolctl 3.5.0, and numpy 2.5.4, scipy
1.18.1, scikit-learn 1.9.1, threadpoolctl 3.8.0. In each environment, the
package's results for the cases of the tree before are byte for byte that
tree's, and its dumps with `--n-threads 2` and `--n-threads 4` hold the bytes
of its one-thread dump.

This is the whole claim of the check: no result is stated for other data,
other versions, other thread counts than those named above, or another
platform. The k-means row above states what its own test compares.

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
