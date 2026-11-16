#!/bin/sh
# Run the suite and the bit-equality check against hmmlearn 0.3.3 in two
# fresh environments, as CI does.
#
# usage: scripts/ci_bit_equality.sh floor|latest WORKDIR
#
# floor:  the three declared minimums and everything else in
#         ci/floor_constraints.txt, binary wheels only, with numpy's AVX-512
#         loops turned off (NPY_DISABLE_CPU_FEATURES).
# latest: the newest binary wheels of numpy, scipy and scikit-learn.
#
# The reference is built from the upstream tag, with the same compiler and
# build settings as the package, so a difference comes from the change and not
# from the build. Both environments get the same Python and the same numpy,
# scipy, scikit-learn and threadpoolctl; `compare` refuses to compare
# otherwise. One thread, one seed. PYTHON names the interpreter (default
# python3); MKVENV is the command that makes an environment with pip in it
# (default "$PYTHON -m venv"); UPSTREAM overrides the repository the reference
# tag is cloned from.
set -eu

cell=${1:?usage: ci_bit_equality.sh floor|latest WORKDIR}
work=${2:?usage: ci_bit_equality.sh floor|latest WORKDIR}
python=${PYTHON:-python3}
mkvenv=${MKVENV:-"$python -m venv"}
upstream=${UPSTREAM:-https://github.com/hmmlearn/hmmlearn}
tag=0.3.3
tag_commit=67c5a95009afe95317923c9fbe75f47962037f48
allow=E1,E2,E3,E4,E5,E6

root=$(cd "$(dirname "$0")/.." && pwd)
case $cell in floor | latest) ;; *) echo "unknown cell: $cell" >&2; exit 2 ;; esac
case $work in /*) ;; *) work=$PWD/$work ;; esac

# On a CPU with AVX-512, numpy before 2.0.2 picks its vector or its scalar
# float64 log loop for a strided input by where the output lands in memory,
# as read from its code. On numpy 1.26.0, on the AVX-512 runners this script
# has run on, the probe below has found values whose strided log, output
# right after the input, differs from the log in place, and none with the
# targets below turned off. The difference is consistent with this cell's
# failures at random: hmmlearn 0.3.3 would then have no single result to
# compare with.
# Off in the floor cell only: the latest cell's numpy has the fix and names its
# targets otherwise. Drop this, the guard and the probe's second run once the
# floor is numpy 2.0.2 or later.
avx512="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX"
avx512="$avx512 AVX512_CNL AVX512_ICL"
if [ "$cell" = floor ]; then
    export NPY_DISABLE_CPU_FEATURES="$avx512"
fi

# One thread and one seed: where a start is seeded, the k-means partial sums
# depend on the order in which threads finish.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HMMCRAFT_SEED=0 PIP_DISABLE_PIP_VERSION_CHECK=1

rm -rf "$work"
mkdir -p "$work"

echo "== reference source: $upstream at $tag"
git clone -q -c advice.detachedHead=false --depth 1 --branch "$tag" "$upstream" "$work/hmmlearn-$tag"
got=$(git -C "$work/hmmlearn-$tag" rev-parse HEAD)
if [ "$got" != "$tag_commit" ]; then
    echo "tag $tag is $got, expected $tag_commit" >&2
    exit 1
fi

for env in fork ref; do
    $mkvenv "$work/$env"
    "$work/$env/bin/python" -m pip install -q --only-binary :all: \
        --require-hashes -r "$root/ci/build_requirements.txt"
done

echo "== runtime dependencies ($cell)"
if [ "$cell" = floor ]; then
    "$work/fork/bin/python" -m pip install -q --only-binary :all: \
        -c "$root/ci/floor_constraints.txt" \
        numpy scipy scikit-learn threadpoolctl pytest
else
    "$work/fork/bin/python" -m pip install -q --only-binary :all: \
        numpy scipy scikit-learn threadpoolctl pytest
fi
"$work/fork/bin/python" -m pip freeze > "$work/fork_freeze.txt"

# The floor cell is only a floor cell if the three minimums are installed.
if [ "$cell" = floor ]; then
    for name in numpy scipy scikit-learn; do
        want=$(grep -i "^$name==" "$root/ci/floor_constraints.txt")
        grep -qix -- "$want" "$work/fork_freeze.txt" \
            || { echo "$want is not what is installed" >&2; exit 1; }
    done
fi

# The reference gets exactly the versions the fork's environment has.
grep -iE '^(numpy|scipy|scikit-learn|threadpoolctl)==' \
    "$work/fork_freeze.txt" > "$work/pins.txt"
"$work/ref/bin/python" -m pip install -q --only-binary :all: \
    -c "$work/pins.txt" numpy scipy scikit-learn threadpoolctl
cat "$work/pins.txt"

echo "== machine"
grep -m1 '^model name' /proc/cpuinfo || true
if grep -qw avx512f /proc/cpuinfo; then
    echo "avx512f: yes"
else
    echo "avx512f: no"
fi
for env in fork ref; do
    echo "-- numpy runtime ($env), as the machine offers it"
    (unset NPY_DISABLE_CPU_FEATURES
     "$work/$env/bin/python" -I -c "import numpy; numpy.show_runtime()")
done

# numpy ignores a name it does not know, with an ImportWarning, and still
# applies the others: a misspelt name would leave its target on unnoticed.
# The guard fails the cell if any AVX-512 dispatch target is still on.
if [ "$cell" = floor ]; then
    for env in fork ref; do
        "$work/$env/bin/python" -I -c "
from numpy.core._multiarray_umath import __cpu_dispatch__, __cpu_features__
on = [k for k in __cpu_dispatch__
      if k.startswith('AVX512') and __cpu_features__[k]]
print('AVX-512 dispatch targets still on ($env):', ' '.join(on) or 'none')
raise SystemExit(1 if on else 0)"
    done
fi

echo "== numpy's float64 log loops"
if [ "$cell" = floor ]; then
    (unset NPY_DISABLE_CPU_FEATURES
     "$work/fork/bin/python" -I "$root/scripts/probe_log_loops.py")
    echo "-- with NPY_DISABLE_CPU_FEATURES=$NPY_DISABLE_CPU_FEATURES"
fi
"$work/fork/bin/python" -I "$root/scripts/probe_log_loops.py"

echo "== build and install"
SETUPTOOLS_SCM_PRETEND_VERSION=$tag "$work/ref/bin/python" -m pip install -q \
    --no-build-isolation --no-deps "$work/hmmlearn-$tag"
(cd "$root" && rm -rf build src/hmm_craft.egg-info)
"$work/fork/bin/python" -m pip install -q --no-build-isolation --no-deps "$root"
(cd "$root" && rm -rf build src/hmm_craft.egg-info)
"$work/fork/bin/python" -I -c \
    "import importlib.metadata as m; print('hmm-craft', m.version('hmm-craft'))"
"$work/ref/bin/python" -I -c \
    "import importlib.metadata as m; print('hmmlearn', m.version('hmmlearn'))"

cd "$root"
echo "== suite"
"$work/fork/bin/python" -I -m pytest -p no:cacheprovider -q \
    --pyargs hmmcraft.tests doc
echo "== the checks' own tests, and the floor declaration"
"$work/fork/bin/python" -I -m pytest -p no:cacheprovider -q \
    scripts/test_bit_equality.py scripts/test_floor_constraints.py \
    scripts/test_check_signoff.py scripts/test_check_pins.py \
    scripts/test_check_ruff_baseline.py

echo "== bit equality"
"$work/ref/bin/python" -I scripts/bit_equality.py dump \
    --package hmmlearn --out "$work/dump_ref"
"$work/fork/bin/python" -I scripts/bit_equality.py dump \
    --package hmmcraft --out "$work/dump_fork" --load-pickles "$work/dump_ref"
"$work/fork/bin/python" -I scripts/bit_equality.py compare \
    "$work/dump_ref" "$work/dump_fork" --allow "$allow"
