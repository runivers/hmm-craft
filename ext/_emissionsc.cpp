// Loops that stand for numpy expressions of the emission code. Each gives
// the bytes of its expression for the arrays it accepts; the Python caller
// decides which arrays those are and keeps numpy for the rest. setup.py
// builds this module with floating-point contraction off, except on Windows,
// where MSVC does not take the option; no loop here has a product that feeds
// a sum, so the results do not depend on it.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using ssize_t = Py_ssize_t;

// The float64 array behind `o`, as it is. Nothing is converted or copied:
// numpy's result depends on the layout it is given, so a conversion here
// would compute another expression than the caller's.
py::array_t<double> as_is(py::object const& o, ssize_t ndim)
{
  if (!py::isinstance<py::array_t<double>>(o)) {
    throw std::invalid_argument{"expected a float64 numpy array"};
  }
  auto a = py::reinterpret_borrow<py::array_t<double>>(o);
  if (a.ndim() != ndim) {
    throw std::invalid_argument{"shape mismatch"};
  }
  auto needed = py::array::c_style | py::detail::npy_api::NPY_ARRAY_ALIGNED_;
  if ((a.flags() & needed) != needed) {
    throw std::invalid_argument{"expected a C-contiguous, aligned array"};
  }
  return a;
}

// numpy's sum of n contiguous doubles (its add reduction along a contiguous
// axis): fewer than 8 terms in order, up to 128 in eight running sums that
// are then added as a tree, more by halves.
double pairwise_sum(double const* a, ssize_t n)
{
  if (n < 8) {
    auto res = -0.;
    for (ssize_t i = 0; i < n; ++i) {
      res += a[i];
    }
    return res;
  }
  if (n <= 128) {
    double r[8];
    for (auto k = 0; k < 8; ++k) {
      r[k] = a[k];
    }
    ssize_t i = 8;
    for (; i < n - n % 8; i += 8) {
      for (auto k = 0; k < 8; ++k) {
        r[k] += a[i + k];
      }
    }
    auto res = ((r[0] + r[1]) + (r[2] + r[3]))
               + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) {
      res += a[i];
    }
    return res;
  }
  auto half = n / 2;
  half -= half % 8;
  return pairwise_sum(a, half) + pairwise_sum(a + half, n - half);
}

// The log-density of each sample under each state's diagonal Gaussian:
// -0.5 * (head + sum over the features of (x - mean)**2 / variance), with
// `head` the caller's term that does not depend on the sample. The sum is
// taken in numpy's order. No product here is added to anything, so the
// result does not depend on contraction. Returns None when a value is not
// finite.
py::object log_density_diag(
  py::object X_, py::object means_, py::object covars_, py::object head_)
{
  auto X = as_is(X_, 2);
  auto means = as_is(means_, 2);
  auto covars = as_is(covars_, 2);
  auto head = as_is(head_, 1);
  auto ns = X.shape(0), nf = X.shape(1), nc = means.shape(0);
  if (means.shape(1) != nf
      || covars.shape(0) != nc || covars.shape(1) != nf
      || head.shape(0) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  if (ns < 1 || nc < 1 || nf < 1) {
    throw std::invalid_argument{"empty array"};
  }
  auto x = X.data(), m = means.data(), v = covars.data(), h = head.data();
  auto log_prob_ = py::array_t<double>{{ns, nc}};
  auto log_prob = log_prob_.mutable_data();
  auto buf = std::vector<double>(nf);
  auto finite = true;
  {
    py::gil_scoped_release nogil;
    for (ssize_t t = 0; finite && t < ns; ++t) {
      for (ssize_t c = 0; finite && c < nc; ++c) {
        for (ssize_t f = 0; f < nf; ++f) {
          auto d = x[t * nf + f] - m[c * nf + f];
          buf[f] = (d * d) / v[c * nf + f];
        }
        auto sum = 0. + pairwise_sum(buf.data(), nf);
        auto value = -0.5 * (h[c] + sum);
        log_prob[t * nc + c] = value;
        finite = std::isfinite(value);
      }
    }
  }
  if (!finite) {
    return py::none();
  }
  return log_prob_;
}

PYBIND11_MODULE(_emissionsc, m) {
  m
    .def("log_density_diag", log_density_diag)
    ;
}
