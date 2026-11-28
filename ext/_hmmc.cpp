#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <atomic>
#include <cfenv>
#include <exception>
#include <limits>
#include <thread>
#include <vector>

namespace py = pybind11;
using ssize_t = Py_ssize_t;

double logaddexp(double a, double b)
{
  return
    a == -std::numeric_limits<double>::infinity() ? b
    : b == -std::numeric_limits<double>::infinity() ? a
    : std::max(a, b) + std::log1p(std::exp(-std::abs(b - a)));
}

double logsumexp(double const* v, ssize_t n)
{
  auto max = *std::max_element(v, v + n);
  if (std::isinf(max)) {
    return max;
  }
  auto acc = 0.;
  for (auto i = 0; i < n; ++i) {
    acc += std::exp(v[i] - max);
  }
  return std::log(acc) + max;
}

py::array_t<double> log(
  py::array_t<double, py::array::c_style | py::array::forcecast> x_)
{
  auto n = x_.size();
  auto ptr = x_.data();
  auto log_ = py::array_t<double>(n);
  auto log_ptr = log_.mutable_data();
  for (auto i = 0; i < n; ++i) {
    *(log_ptr++) = std::log(*(ptr++));
  }
  if (std::fetestexcept(FE_DIVBYZERO)) {
    std::feclearexcept(FE_DIVBYZERO);  // log(0) = -inf, ignore exception.
  }
  return log_.reshape(
    std::vector<ssize_t>(x_.shape(), x_.shape() + x_.ndim()));
}

// Most recursions start at row 0 or row ns - 1 of a lattice, and the
// log-space ones take a maximum over the states, through unchecked
// accessors: with no rows, or no states, they read or write outside the
// arrays. Every function checks, so none depends on how its caller built
// the input.
void check_not_empty(ssize_t ns, ssize_t nc)
{
  if (ns < 1) {
    throw std::invalid_argument{"a sequence needs at least one sample"};
  }
  if (nc < 1) {
    throw std::invalid_argument{"a model needs at least one state"};
  }
}

std::tuple<double, py::array_t<double>, py::array_t<double>> forward_scaling(
  py::array_t<double> startprob_,
  py::array_t<double> transmat_,
  py::array_t<double> frameprob_)
{
  auto min_sum = 1e-300;

  auto startprob = startprob_.unchecked<1>();
  auto transmat = transmat_.unchecked<2>();
  auto frameprob = frameprob_.unchecked<2>();
  auto ns = frameprob.shape(0), nc = frameprob.shape(1);
  if (startprob.shape(0) != nc
      || transmat.shape(0) != nc || transmat.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  check_not_empty(ns, nc);
  auto fwdlattice_ = py::array_t<double>{{ns, nc}};
  auto fwd = fwdlattice_.mutable_unchecked<2>();
  auto scaling_ = py::array_t<double>(ns);
  auto scaling = scaling_.mutable_unchecked<1>();
  auto log_prob = 0.;
  {
    py::gil_scoped_release nogil;
    std::fill_n(fwd.mutable_data(0, 0), fwd.size(), 0);
    for (auto i = 0; i < nc; ++i) {
      fwd(0, i) = startprob(i) * frameprob(0, i);
    }
    auto sum0 = std::accumulate(&fwd(0, 0), &fwd(0, nc), 0.);
    if (sum0 < min_sum) {
      throw std::range_error{"forward pass failed with underflow; "
                             "consider using implementation='log' instead"};
    }
    auto scale0 = scaling(0) = 1. / sum0;
    log_prob -= std::log(scale0);
    for (auto i = 0; i < nc; ++i) {
      fwd(0, i) *= scale0;
    }
    for (auto t = 1; t < ns; ++t) {
      for (auto j = 0; j < nc; ++j) {
        for (auto i = 0; i < nc; ++i) {
          fwd(t, j) += fwd(t - 1, i) * transmat(i, j);
        }
        fwd(t, j) *= frameprob(t, j);
      }
      auto sum = std::accumulate(&fwd(t, 0), &fwd(t, nc), 0.);
      if (sum < min_sum) {
        throw std::range_error{"forward pass failed with underflow; "
                               "consider using implementation='log' instead"};
      }
      auto scale = scaling(t) = 1. / sum;
      log_prob -= std::log(scale);
      for (auto j = 0; j < nc; ++j) {
        fwd(t, j) *= scale;
      }
    }
  }
  return {log_prob, fwdlattice_, scaling_};
}

std::tuple<double, py::array_t<double>> forward_log(
  py::array_t<double> startprob_,
  py::array_t<double> transmat_,
  py::array_t<double> log_frameprob_)
{
  auto log_startprob_ = log(startprob_);
  auto log_startprob = log_startprob_.unchecked<1>();
  auto log_transmat_ = log(transmat_);
  auto log_transmat = log_transmat_.unchecked<2>();
  auto log_frameprob = log_frameprob_.unchecked<2>();
  auto ns = log_frameprob.shape(0), nc = log_frameprob.shape(1);
  if (log_startprob.shape(0) != nc
      || log_transmat.shape(0) != nc || log_transmat.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  check_not_empty(ns, nc);
  auto buf = std::vector<double>(nc);
  auto fwdlattice_ = py::array_t<double>{{ns, nc}};
  auto fwd = fwdlattice_.mutable_unchecked<2>();
  {
    py::gil_scoped_release nogil;
    for (auto i = 0; i < nc; ++i) {
      fwd(0, i) = log_startprob(i) + log_frameprob(0, i);
    }
    for (auto t = 1; t < ns; ++t) {
      for (auto j = 0; j < nc; ++j) {
        for (auto i = 0; i < nc; ++i) {
          buf[i] = fwd(t - 1, i) + log_transmat(i, j);
        }
        fwd(t, j) = logsumexp(buf.data(), nc) + log_frameprob(t, j);
      }
    }
  }
  auto log_prob = logsumexp(&fwd(ns - 1, 0), nc);
  return {log_prob, fwdlattice_};
}

py::array_t<double> backward_scaling(
  py::array_t<double> startprob_,
  py::array_t<double> transmat_,
  py::array_t<double> frameprob_,
  py::array_t<double> scaling_)
{
  auto startprob = startprob_.unchecked<1>();
  auto transmat = transmat_.unchecked<2>();
  auto frameprob = frameprob_.unchecked<2>();
  auto scaling = scaling_.unchecked<1>();
  auto ns = frameprob.shape(0), nc = frameprob.shape(1);
  if (startprob.shape(0) != nc
      || transmat.shape(0) != nc || transmat.shape(1) != nc
      || scaling.shape(0) != ns) {
    throw std::invalid_argument{"shape mismatch"};
  }
  check_not_empty(ns, nc);
  auto bwdlattice_ = py::array_t<double>{{ns, nc}};
  auto bwd = bwdlattice_.mutable_unchecked<2>();
  py::gil_scoped_release nogil;
  std::fill_n(bwd.mutable_data(0, 0), bwd.size(), 0);
  for (auto i = 0; i < nc; ++i) {
    bwd(ns - 1, i) = scaling(ns - 1);
  }
  for (auto t = ns - 2; t >= 0; --t) {
    for (auto i = 0; i < nc; ++i) {
      for (auto j = 0; j < nc; ++j) {
        bwd(t, i) += transmat(i, j) * frameprob(t + 1, j) * bwd(t + 1, j);
      }
      bwd(t, i) *= scaling(t);
    }
  }
  return bwdlattice_;
}

py::array_t<double> backward_log(
  py::array_t<double> startprob_,
  py::array_t<double> transmat_,
  py::array_t<double> log_frameprob_)
{
  auto log_startprob_ = log(startprob_);
  auto log_startprob = log_startprob_.unchecked<1>();
  auto log_transmat_ = log(transmat_);
  auto log_transmat = log_transmat_.unchecked<2>();
  auto log_frameprob = log_frameprob_.unchecked<2>();
  auto ns = log_frameprob.shape(0), nc = log_frameprob.shape(1);
  if (log_startprob.shape(0) != nc
      || log_transmat.shape(0) != nc || log_transmat.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  check_not_empty(ns, nc);
  auto buf = std::vector<double>(nc);
  auto bwdlattice_ = py::array_t<double>{{ns, nc}};
  auto bwd = bwdlattice_.mutable_unchecked<2>();
  py::gil_scoped_release nogil;
  for (auto i = 0; i < nc; ++i) {
    bwd(ns - 1, i) = 0;
  }
  for (auto t = ns - 2; t >= 0; --t) {
    for (auto i = 0; i < nc; ++i) {
      for (auto j = 0; j < nc; ++j) {
        buf[j] = log_transmat(i, j) + log_frameprob(t + 1, j) + bwd(t + 1, j);
      }
      bwd(t, i) = logsumexp(buf.data(), nc);
    }
  }
  return bwdlattice_;
}

py::array_t<double> compute_scaling_xi_sum(
  py::array_t<double> fwdlattice_,
  py::array_t<double> transmat_,
  py::array_t<double> bwdlattice_,
  py::array_t<double> frameprob_)
{
  auto fwd = fwdlattice_.unchecked<2>();
  auto transmat = transmat_.unchecked<2>();
  auto bwd = bwdlattice_.unchecked<2>();
  auto frameprob = frameprob_.unchecked<2>();
  auto ns = frameprob.shape(0), nc = frameprob.shape(1);
  if (fwd.shape(0) != ns || fwd.shape(1) != nc
      || transmat.shape(0) != nc || transmat.shape(1) != nc
      || bwd.shape(0) != ns || bwd.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  check_not_empty(ns, nc);
  auto xi_sum_ = py::array_t<double>{{nc, nc}};
  auto xi_sum = xi_sum_.mutable_unchecked<2>();
  std::fill_n(xi_sum.mutable_data(0, 0), xi_sum.size(), 0);
  py::gil_scoped_release nogil;
  for (auto t = 0; t < ns - 1; ++t) {
    for (auto i = 0; i < nc; ++i) {
      for (auto j = 0; j < nc; ++j) {
        xi_sum(i, j) += fwd(t, i)
                        * transmat(i, j)
                        * frameprob(t + 1, j)
                        * bwd(t + 1, j);
      }
    }
  }
  return xi_sum_;
}

py::array_t<double> compute_log_xi_sum(
  py::array_t<double> fwdlattice_,
  py::array_t<double> transmat_,
  py::array_t<double> bwdlattice_,
  py::array_t<double> log_frameprob_)
{
  auto fwd = fwdlattice_.unchecked<2>();
  auto log_transmat_ = log(transmat_);
  auto log_transmat = log_transmat_.unchecked<2>();
  auto bwd = bwdlattice_.unchecked<2>();
  auto log_frameprob = log_frameprob_.unchecked<2>();
  auto ns = log_frameprob.shape(0), nc = log_frameprob.shape(1);
  if (fwd.shape(0) != ns || fwd.shape(1) != nc
      || log_transmat.shape(0) != nc || log_transmat.shape(1) != nc
      || bwd.shape(0) != ns || bwd.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  check_not_empty(ns, nc);
  auto log_prob = logsumexp(&fwd(ns - 1, 0), nc);
  auto log_xi_sum_ = py::array_t<double>{{nc, nc}};
  auto log_xi_sum = log_xi_sum_.mutable_unchecked<2>();
  std::fill_n(log_xi_sum.mutable_data(0, 0), log_xi_sum.size(),
              -std::numeric_limits<double>::infinity());
  py::gil_scoped_release nogil;
  for (auto t = 0; t < ns - 1; ++t) {
    for (auto i = 0; i < nc; ++i) {
      for (auto j = 0; j < nc; ++j) {
        auto log_xi = fwd(t, i)
                      + log_transmat(i, j)
                      + log_frameprob(t + 1, j)
                      + bwd(t + 1, j)
                      - log_prob;
        log_xi_sum(i, j) = logaddexp(log_xi_sum(i, j), log_xi);
      }
    }
  }
  return log_xi_sum_;
}

std::tuple<double, py::array_t<ssize_t>> viterbi(
  py::array_t<double> startprob_,
  py::array_t<double> transmat_,
  py::array_t<double> log_frameprob_)
{
  auto log_startprob_ = log(startprob_);
  auto log_startprob = log_startprob_.unchecked<1>();
  auto log_transmat_ = log(transmat_);
  auto log_transmat = log_transmat_.unchecked<2>();
  auto log_frameprob = log_frameprob_.unchecked<2>();
  auto ns = log_frameprob.shape(0), nc = log_frameprob.shape(1);
  if (log_startprob.shape(0) != nc
      || log_transmat.shape(0) != nc || log_transmat.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  check_not_empty(ns, nc);
  auto state_sequence_ = py::array_t<ssize_t>(ns);
  auto viterbi_lattice_ = py::array_t<double>{{ns, nc}};
  auto state_sequence = state_sequence_.mutable_unchecked<1>();
  auto viterbi_lattice = viterbi_lattice_.mutable_unchecked<2>();
  {
    py::gil_scoped_release nogil;
    for (auto i = 0; i < nc; ++i) {
      viterbi_lattice(0, i) = log_startprob(i) + log_frameprob(0, i);
    }
    for (auto t = 1; t < ns; ++t) {
      for (auto i = 0; i < nc; ++i) {
        auto max = -std::numeric_limits<double>::infinity();
        for (auto j = 0; j < nc; ++j) {
          max = std::max(max, viterbi_lattice(t - 1, j) + log_transmat(j, i));
        }
        viterbi_lattice(t, i) = max + log_frameprob(t, i);
      }
    }
    auto row = &viterbi_lattice(ns - 1, 0);
    auto prev = state_sequence(ns - 1) = std::max_element(row, row + nc) - row;
    for (auto t = ns - 2; t >= 0; --t) {
      auto max = std::make_pair(-std::numeric_limits<double>::infinity(), 0);
      for (auto i = 0; i < nc; ++i) {
        max = std::max(max, {viterbi_lattice(t, i) + log_transmat(i, prev), i});
      }
      state_sequence(t) = prev = max.second;
    }
  }
  return {viterbi_lattice(ns - 1, state_sequence(ns - 1)), state_sequence_};
}

// The block functions run, for each sequence of a block, the statements of
// forward_*, backward_* and compute_*_xi_sum above, in the same order and
// with the same operands, on rows of a matrix that stacks the sequences'
// lattices.  They read through pointers, so every array they take is
// converted to a C-contiguous one first.

using c_array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using c_lengths =
  py::array_t<ssize_t, py::array::c_style | py::array::forcecast>;

// Checks the shapes and the lengths, which count rows of the lattice, and
// returns the longest length.
ssize_t check_block(
  c_array const& startprob_,
  c_array const& transmat_,
  c_array const& lattice_,
  c_lengths const& lengths_)
{
  if (startprob_.ndim() != 1 || transmat_.ndim() != 2
      || lattice_.ndim() != 2 || lengths_.ndim() != 1
      || startprob_.shape(0) != lattice_.shape(1)
      || transmat_.shape(0) != lattice_.shape(1)
      || transmat_.shape(1) != lattice_.shape(1)) {
    throw std::invalid_argument{"shape mismatch"};
  }
  auto ns = lattice_.shape(0), nc = lattice_.shape(1);
  check_not_empty(ns, nc);
  auto lengths = lengths_.data();
  auto total = ssize_t{0}, longest = ssize_t{0};
  for (auto s = ssize_t{0}; s < lengths_.shape(0); ++s) {
    if (lengths[s] < 1) {
      throw std::invalid_argument{"a sequence needs at least one sample"};
    }
    if (lengths[s] > ns - total) {  // Not total + length: no overflow.
      throw std::invalid_argument{"lengths do not sum to the lattice rows"};
    }
    total += lengths[s];
    longest = std::max(longest, lengths[s]);
  }
  if (total != ns) {
    throw std::invalid_argument{"lengths do not sum to the lattice rows"};
  }
  return longest;
}

// forward_log, backward_log and compute_log_xi_sum for one sequence of ns
// rows; xi is null when the transition sums are not wanted.
double estep_log_sequence(
  double const* log_startprob, double const* log_transmat,
  double const* log_frameprob, ssize_t ns, ssize_t nc, double* buf,
  double* fwd, double* bwd, double* xi)
{
  for (auto i = ssize_t{0}; i < nc; ++i) {
    fwd[i] = log_startprob[i] + log_frameprob[i];
  }
  for (auto t = ssize_t{1}; t < ns; ++t) {
    for (auto j = ssize_t{0}; j < nc; ++j) {
      for (auto i = ssize_t{0}; i < nc; ++i) {
        buf[i] = fwd[(t - 1) * nc + i] + log_transmat[i * nc + j];
      }
      fwd[t * nc + j] = logsumexp(buf, nc) + log_frameprob[t * nc + j];
    }
  }
  auto log_prob = logsumexp(fwd + (ns - 1) * nc, nc);
  for (auto i = ssize_t{0}; i < nc; ++i) {
    bwd[(ns - 1) * nc + i] = 0;
  }
  for (auto t = ns - 2; t >= 0; --t) {
    for (auto i = ssize_t{0}; i < nc; ++i) {
      for (auto j = ssize_t{0}; j < nc; ++j) {
        buf[j] = log_transmat[i * nc + j]
                 + log_frameprob[(t + 1) * nc + j]
                 + bwd[(t + 1) * nc + j];
      }
      bwd[t * nc + i] = logsumexp(buf, nc);
    }
  }
  if (xi) {
    std::fill_n(xi, nc * nc, -std::numeric_limits<double>::infinity());
    // Taken again, as compute_log_xi_sum takes it.
    auto xi_log_prob = logsumexp(fwd + (ns - 1) * nc, nc);
    for (auto t = ssize_t{0}; t < ns - 1; ++t) {
      for (auto i = ssize_t{0}; i < nc; ++i) {
        for (auto j = ssize_t{0}; j < nc; ++j) {
          auto log_xi = fwd[t * nc + i]
                        + log_transmat[i * nc + j]
                        + log_frameprob[(t + 1) * nc + j]
                        + bwd[(t + 1) * nc + j]
                        - xi_log_prob;
          xi[i * nc + j] = logaddexp(xi[i * nc + j], log_xi);
        }
      }
    }
  }
  return log_prob;
}

// forward_scaling, backward_scaling and compute_scaling_xi_sum for one
// sequence of ns rows; xi is null when the transition sums are not wanted.
// Returns false, with the results partly written, where forward_scaling
// throws its underflow error.
bool estep_scaling_sequence(
  double const* startprob, double const* transmat, double const* frameprob,
  ssize_t ns, ssize_t nc, double* scaling, double* log_prob_out,
  double* fwd, double* bwd, double* xi)
{
  auto min_sum = 1e-300;
  auto log_prob = 0.;
  std::fill_n(fwd, ns * nc, 0);
  for (auto i = ssize_t{0}; i < nc; ++i) {
    fwd[i] = startprob[i] * frameprob[i];
  }
  auto sum0 = std::accumulate(fwd, fwd + nc, 0.);
  if (sum0 < min_sum) {
    return false;
  }
  auto scale0 = scaling[0] = 1. / sum0;
  log_prob -= std::log(scale0);
  for (auto i = ssize_t{0}; i < nc; ++i) {
    fwd[i] *= scale0;
  }
  for (auto t = ssize_t{1}; t < ns; ++t) {
    for (auto j = ssize_t{0}; j < nc; ++j) {
      for (auto i = ssize_t{0}; i < nc; ++i) {
        fwd[t * nc + j] += fwd[(t - 1) * nc + i] * transmat[i * nc + j];
      }
      fwd[t * nc + j] *= frameprob[t * nc + j];
    }
    auto sum = std::accumulate(fwd + t * nc, fwd + (t + 1) * nc, 0.);
    if (sum < min_sum) {
      return false;
    }
    auto scale = scaling[t] = 1. / sum;
    log_prob -= std::log(scale);
    for (auto j = ssize_t{0}; j < nc; ++j) {
      fwd[t * nc + j] *= scale;
    }
  }
  std::fill_n(bwd, ns * nc, 0);
  for (auto i = ssize_t{0}; i < nc; ++i) {
    bwd[(ns - 1) * nc + i] = scaling[ns - 1];
  }
  for (auto t = ns - 2; t >= 0; --t) {
    for (auto i = ssize_t{0}; i < nc; ++i) {
      for (auto j = ssize_t{0}; j < nc; ++j) {
        bwd[t * nc + i] += transmat[i * nc + j]
                           * frameprob[(t + 1) * nc + j]
                           * bwd[(t + 1) * nc + j];
      }
      bwd[t * nc + i] *= scaling[t];
    }
  }
  if (xi) {
    std::fill_n(xi, nc * nc, 0);
    for (auto t = ssize_t{0}; t < ns - 1; ++t) {
      for (auto i = ssize_t{0}; i < nc; ++i) {
        for (auto j = ssize_t{0}; j < nc; ++j) {
          xi[i * nc + j] += fwd[t * nc + i]
                            * transmat[i * nc + j]
                            * frameprob[(t + 1) * nc + j]
                            * bwd[(t + 1) * nc + j];
        }
      }
    }
  }
  *log_prob_out = log_prob;
  return true;
}

// Calls work(s, k) once for every sequence s of [0, n), on n_threads
// threads; k in [0, n_threads) tells the calling thread (0) and the
// threads started here apart, so that each uses its own scratch.  Each
// thread takes the next sequence not yet taken, so which thread computes a
// sequence is not fixed; work(s, k) must write nothing another sequence
// reads.  The threads started here compute in the calling thread's
// floating-point environment, and the exception flags they raise are raised
// on the calling thread before returning, as if it had computed every
// sequence.  When the system refuses a thread, the sequences are computed
// by the threads that started and by the calling thread.  An exception
// thrown by work is rethrown here once every thread has ended.
template <typename Work>
void for_each_sequence(ssize_t n, ssize_t n_threads, Work const& work)
{
  std::atomic<ssize_t> next{0};  // Not "auto next = ...": no copy in C++11.
  auto errors = std::vector<std::exception_ptr>(n_threads);
  auto raised = std::vector<int>(n_threads, 0);
  std::fenv_t env;
  std::fegetenv(&env);
  auto run = [&](ssize_t k) {
    try {
      if (k) {
        std::fesetenv(&env);
      }
      for (auto s = next.fetch_add(1); s < n; s = next.fetch_add(1)) {
        work(s, k);
      }
    } catch (...) {
      errors[k] = std::current_exception();
      next.store(n);  // The other threads stop after their sequence.
    }
    if (k) {
      raised[k] = std::fetestexcept(FE_ALL_EXCEPT);
    }
  };
  auto threads = std::vector<std::thread>{};
  threads.reserve(n_threads - 1);  // No reallocation once one runs.
  try {
    for (auto k = ssize_t{1}; k < n_threads; ++k) {
      threads.emplace_back(run, k);
    }
  } catch (std::exception const&) {
    // std::system_error: the system refuses another thread; std::bad_alloc:
    // no memory for its state.  The sequences are computed all the same.
  }
  run(0);
  auto flags = 0;
  for (auto k = size_t{0}; k < threads.size(); ++k) {
    threads[k].join();
    flags |= raised[k + 1];
  }
  if (flags) {
    std::feraiseexcept(flags);
  }
  for (auto const& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
}

void check_threads(ssize_t n_threads)
{
  if (n_threads < 1) {
    throw std::invalid_argument{"n_threads must be at least 1"};
  }
}

// The first row of each sequence in the stacked lattice.
std::vector<ssize_t> first_rows(ssize_t const* lengths, ssize_t nq)
{
  auto first = std::vector<ssize_t>(nq);
  auto row = ssize_t{0};
  for (auto s = ssize_t{0}; s < nq; ++s) {
    first[s] = row;
    row += lengths[s];
  }
  return first;
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>,
           py::array_t<double>> estep_log(
  c_array startprob_,
  c_array transmat_,
  c_array log_frameprob_,
  c_lengths lengths_,
  bool with_xi_sum,
  ssize_t n_threads)
{
  check_block(startprob_, transmat_, log_frameprob_, lengths_);
  check_threads(n_threads);
  auto ns = log_frameprob_.shape(0), nc = log_frameprob_.shape(1);
  auto nq = lengths_.shape(0);
  auto threads = std::min(n_threads, nq);
  auto log_startprob_ = log(startprob_);
  auto log_transmat_ = log(transmat_);
  auto buf = std::vector<double>(nc);
  auto log_prob_ = py::array_t<double>(nq);
  auto fwdlattice_ = py::array_t<double>{{ns, nc}};
  auto bwdlattice_ = py::array_t<double>{{ns, nc}};
  auto log_xi_sum_ = py::array_t<double>{{with_xi_sum ? nq : 0, nc, nc}};
  auto lengths = lengths_.data();
  auto log_startprob = log_startprob_.data();
  auto log_transmat = log_transmat_.data();
  auto log_frameprob = log_frameprob_.data();
  auto log_prob = log_prob_.mutable_data();
  auto fwd = fwdlattice_.mutable_data();
  auto bwd = bwdlattice_.mutable_data();
  auto xi = with_xi_sum ? log_xi_sum_.mutable_data() : nullptr;
  {
    py::gil_scoped_release nogil;
    if (threads == 1) {
      auto row = ssize_t{0};
      for (auto s = ssize_t{0}; s < nq; ++s) {
        log_prob[s] = estep_log_sequence(
          log_startprob, log_transmat, log_frameprob + row * nc, lengths[s],
          nc, buf.data(), fwd + row * nc, bwd + row * nc,
          xi ? xi + s * nc * nc : nullptr);
        row += lengths[s];
      }
    } else {
      auto first = first_rows(lengths, nq);
      // A cache line between the scratch of two threads.
      auto stride = nc + 8;
      auto bufs = std::vector<double>(threads * stride);
      for_each_sequence(nq, threads, [&](ssize_t s, ssize_t k) {
        auto row = first[s];
        log_prob[s] = estep_log_sequence(
          log_startprob, log_transmat, log_frameprob + row * nc, lengths[s],
          nc, bufs.data() + k * stride, fwd + row * nc, bwd + row * nc,
          xi ? xi + s * nc * nc : nullptr);
      });
    }
  }
  return std::make_tuple(log_prob_, fwdlattice_, bwdlattice_, log_xi_sum_);
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>,
           py::array_t<double>, py::array_t<bool>> estep_scaling(
  c_array startprob_,
  c_array transmat_,
  c_array frameprob_,
  c_lengths lengths_,
  bool with_xi_sum,
  ssize_t n_threads)
{
  auto longest = check_block(startprob_, transmat_, frameprob_, lengths_);
  check_threads(n_threads);
  auto ns = frameprob_.shape(0), nc = frameprob_.shape(1);
  auto nq = lengths_.shape(0);
  auto threads = std::min(n_threads, nq);
  auto scaling = std::vector<double>(longest);
  auto log_prob_ = py::array_t<double>(nq);
  auto fwdlattice_ = py::array_t<double>{{ns, nc}};
  auto bwdlattice_ = py::array_t<double>{{ns, nc}};
  auto xi_sum_ = py::array_t<double>{{with_xi_sum ? nq : 0, nc, nc}};
  auto underflow_ = py::array_t<bool>(nq);
  auto lengths = lengths_.data();
  auto startprob = startprob_.data();
  auto transmat = transmat_.data();
  auto frameprob = frameprob_.data();
  auto log_prob = log_prob_.mutable_data();
  auto fwd = fwdlattice_.mutable_data();
  auto bwd = bwdlattice_.mutable_data();
  auto xi = with_xi_sum ? xi_sum_.mutable_data() : nullptr;
  auto underflow = underflow_.mutable_data();
  {
    py::gil_scoped_release nogil;
    auto quiet_nan = std::numeric_limits<double>::quiet_NaN();
    auto sequence = [&](ssize_t s, ssize_t row, double* scratch) {
      auto n = lengths[s];
      auto xi_s = xi ? xi + s * nc * nc : nullptr;
      underflow[s] = !estep_scaling_sequence(
        startprob, transmat, frameprob + row * nc, n, nc, scratch,
        log_prob + s, fwd + row * nc, bwd + row * nc, xi_s);
      if (underflow[s]) {
        // forward_scaling would throw: NaN in every result of the sequence,
        // so that a result used by mistake shows.
        log_prob[s] = quiet_nan;
        std::fill_n(fwd + row * nc, n * nc, quiet_nan);
        std::fill_n(bwd + row * nc, n * nc, quiet_nan);
        if (xi_s) {
          std::fill_n(xi_s, nc * nc, quiet_nan);
        }
      }
    };
    if (threads == 1) {
      auto row = ssize_t{0};
      for (auto s = ssize_t{0}; s < nq; ++s) {
        sequence(s, row, scaling.data());
        row += lengths[s];
      }
    } else {
      auto first = first_rows(lengths, nq);
      auto stride = longest + 8;
      auto scalings = std::vector<double>(threads * stride);
      for_each_sequence(nq, threads, [&](ssize_t s, ssize_t k) {
        sequence(s, first[s], scalings.data() + k * stride);
      });
    }
  }
  return std::make_tuple(
    log_prob_, fwdlattice_, bwdlattice_, xi_sum_, underflow_);
}

PYBIND11_MODULE(_hmmc, m) {
  m
    .def("forward_scaling", forward_scaling)
    .def("forward_log", forward_log)
    .def("backward_scaling", backward_scaling)
    .def("backward_log", backward_log)
    .def("compute_scaling_xi_sum", compute_scaling_xi_sum)
    .def("compute_log_xi_sum", compute_log_xi_sum)
    .def("viterbi", viterbi)
    .def("estep_log", estep_log,
         py::arg("startprob"), py::arg("transmat"), py::arg("log_frameprob"),
         py::arg("lengths"), py::arg("with_xi_sum"), py::arg("n_threads") = 1)
    .def("estep_scaling", estep_scaling,
         py::arg("startprob"), py::arg("transmat"), py::arg("frameprob"),
         py::arg("lengths"), py::arg("with_xi_sum"), py::arg("n_threads") = 1)
    ;
}
