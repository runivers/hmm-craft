#!/bin/sh
# Measure what the compiler option -ffp-contract=off changes on this machine.
# It reports; it is not a gate.
#
# usage: scripts/measure_contraction.sh floor|latest WORKDIR
#
# Three builds with one compiler, each in its own environment with the same
# numpy, scipy, scikit-learn and threadpoolctl (floor and latest as in
# ci_bit_equality.sh):
#   REF      hmmlearn 0.3.3 from its tag, as its setup.py builds it
#   REF_OFF  the same source with the option added
#   FORK     this package, as its setup.py builds it: its kernel, like REF,
#            without the option
# and three comparisons of scripts/bit_equality.py dumps:
#   A  REF against REF_OFF, no exception allowed: what the option changes in
#      hmmlearn 0.3.3's results on this machine
#   B  REF against FORK, the gate's exceptions: this package against
#      hmmlearn 0.3.3 as upstream builds it, the option on neither side, so
#      a difference here does not come from the option
#   C  REF_OFF against FORK, the gate's exceptions: the option on one side
#      only, so C is expected to differ where A does, outside the arrays
#      those exceptions cover
# scripts/probe_contraction.py then reports, per build, whether the kernel
# fuses a product into a sum, and what numpy does. It reports the same for the
# kernels of two published wheels for this platform, installed in a fourth
# environment: hmmlearn 0.3.3 and this package 0.4.1, both built without
# the option, as their users have them. The wheels' file names and sha256
# are in out/wheels.log.
#
# setuptools puts CXXFLAGS in place of the interpreter's flags on the C++
# compile line and appends CPPFLAGS to it, so the option goes in CPPFLAGS:
# REF_OFF differs from REF by the option alone. A CPPFLAGS already in the
# environment reaches all three builds. Each build starts from a tree without
# build/, or setuptools reuses the object file of the build before it.
#
# The tag, its commit and the exception list are read from ci_bit_equality.sh.
# PYTHON, MKVENV and UPSTREAM as there. No step stops the script. Everything
# is kept under WORKDIR/out; WORKDIR/out/summary.txt says how to read it.
# Exit status: 0 when every step gave a result (a comparison that finds
# differences is a result), 1 when one did not, 2 for a wrong call.
set -u

cell=${1:?usage: measure_contraction.sh floor|latest WORKDIR}
work=${2:?usage: measure_contraction.sh floor|latest WORKDIR}
python=${PYTHON:-python3}
mkvenv=${MKVENV:-"$python -m venv"}
upstream=${UPSTREAM:-https://github.com/hmmlearn/hmmlearn}
option=-ffp-contract=off
cppflags=${CPPFLAGS:-}
# A release of this package built without the option, on PyPI.
published=0.4.1

root=$(cd "$(dirname "$0")/.." && pwd)
gate=$root/scripts/ci_bit_equality.sh
tag=$(sed -n 's/^tag=//p' "$gate")
tag_commit=$(sed -n 's/^tag_commit=//p' "$gate")
allow=$(sed -n 's/^allow=//p' "$gate")
case $cell in floor | latest) ;; *) echo "unknown cell: $cell" >&2; exit 2 ;; esac
if [ -z "$tag" ] || [ -z "$tag_commit" ] || [ -z "$allow" ]; then
    echo "cannot read tag, tag_commit and allow from $gate" >&2
    exit 2
fi
case $work in /*) ;; *) work=$PWD/$work ;; esac
out=$work/out
src=$work/hmmlearn-$tag
check=$root/scripts/bit_equality.py
probe=$root/scripts/probe_contraction.py
site_lines='^(forward_scaling|backward_scaling|compute_scaling_xi_sum): '

# As in ci_bit_equality.sh: on x86_64, numpy before 2.0.2 leaves the full and
# tied log-densities to memory placement unless its AVX-512 loops are off.
avx512="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX"
avx512="$avx512 AVX512_CNL AVX512_ICL"
if [ "$cell" = floor ] && [ "$(uname -m)" = x86_64 ]; then
    export NPY_DISABLE_CPU_FEATURES="$avx512"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HMMCRAFT_SEED=0 PIP_DISABLE_PIP_VERSION_CHECK=1

rm -rf "$work"
mkdir -p "$out"
: > "$out/status.txt"

# step NAME COMMAND...: run, show and keep the output, record the status.
step() {
    step_name=$1
    shift
    echo "== $step_name"
    "$@" > "$out/$step_name.log" 2>&1
    step_status=$?
    cat "$out/$step_name.log"
    echo "$step_name $step_status" >> "$out/status.txt"
}

clone() {
    git clone -q -c advice.detachedHead=false --depth 1 --branch "$tag" \
        "$upstream" "$src" || return 1
    got=$(git -C "$src" rev-parse HEAD)
    if [ "$got" != "$tag_commit" ]; then
        echo "tag $tag is $got, expected $tag_commit"
        return 1
    fi
}

environments() {
    for env in fork ref ref_off; do
        $mkvenv "$work/$env" || return 1
        "$work/$env/bin/python" -m pip install -q --only-binary :all: \
            --require-hashes -r "$root/ci/build_requirements.txt" || return 1
    done
    if [ "$cell" = floor ]; then
        "$work/fork/bin/python" -m pip install -q --only-binary :all: \
            -c "$root/ci/floor_constraints.txt" \
            numpy scipy scikit-learn threadpoolctl || return 1
    else
        "$work/fork/bin/python" -m pip install -q --only-binary :all: \
            numpy scipy scikit-learn threadpoolctl || return 1
    fi
    "$work/fork/bin/python" -m pip freeze > "$out/fork_freeze.txt" || return 1
    grep -iE '^(numpy|scipy|scikit-learn|threadpoolctl)==' \
        "$out/fork_freeze.txt" > "$out/pins.txt" || return 1
    for env in ref ref_off; do
        "$work/$env/bin/python" -m pip install -q --only-binary :all: \
            -c "$out/pins.txt" numpy scipy scikit-learn threadpoolctl \
            || return 1
    done
    cat "$out/pins.txt"
}

# The published wheels for this platform, each kept in out/wheels and
# installed from that file; pip picks the wheel as for a user.
wheels() {
    $mkvenv "$work/wheels" || return 1
    "$work/wheels/bin/python" -m pip install -q --only-binary :all: \
        -c "$out/pins.txt" numpy scipy scikit-learn threadpoolctl \
        || return 1
    mkdir -p "$out/wheels"
    "$work/wheels/bin/python" -m pip download -q --only-binary :all: \
        --no-deps -d "$out/wheels" "hmmlearn==$tag" \
        "hmm-craft==$published" || return 1
    # Python's hashlib: neither sha256sum nor shasum is on every system.
    "$work/wheels/bin/python" -I -c "
import hashlib
import sys

for path in sorted(sys.argv[1:]):
    with open(path, 'rb') as file:
        digest = hashlib.sha256(file.read()).hexdigest()
    print(path.rsplit('/', 1)[-1], digest)
" "$out/wheels"/*.whl || return 1
    "$work/wheels/bin/python" -m pip install -q --no-index --no-deps \
        "$out/wheels"/*.whl || return 1
    "$work/wheels/bin/python" -I -c "
import importlib.metadata as m

for name in 'hmmlearn', 'hmm-craft':
    print(name, m.version(name))
"
}

# probe_kernel ENV PACKAGE: the kernel's three sites; no result unless the
# probe printed a line for each.
probe_kernel() {
    "$work/$1/bin/python" -I "$probe" --package "$2" > "$out/probe.tmp"
    probe_status=$?
    cat "$out/probe.tmp"
    sites=$(grep -cE "$site_lines" "$out/probe.tmp")
    rm -f "$out/probe.tmp"
    [ "$probe_status" = 0 ] && [ "$sites" = 3 ]
}

# build_reference ENV FLAGS: hmmlearn from a tree without build/.
build_reference() {
    rm -rf "$src/build"
    CPPFLAGS=$2 SETUPTOOLS_SCM_PRETEND_VERSION=$tag \
        "$work/$1/bin/python" -m pip install -v --no-build-isolation \
        --no-deps "$src"
}

build_fork() {
    (cd "$root" && rm -rf build src/hmm_craft.egg-info)
    CPPFLAGS=$cppflags "$work/fork/bin/python" -m pip install -v \
        --no-build-isolation --no-deps "$root"
    build_status=$?
    (cd "$root" && rm -rf build src/hmm_craft.egg-info)
    return "$build_status"
}

# compile_lines LOG [SOURCE]: the compiler commands for SOURCE in a build
# log, by default the kernel, ext/_hmmc.cpp.
compile_lines() {
    grep -- " -c ${2:-ext/_hmmc.cpp}" "$1"
}

# option_count LOG [SOURCE]: "<lines with the option> of <compile lines>".
option_count() {
    echo "$(compile_lines "$1" "${2:-ext/_hmmc.cpp}" | grep -c -- "$option") of" \
        "$(compile_lines "$1" "${2:-ext/_hmmc.cpp}" | awk 'END {print NR}')"
}

# The builds are what their names say: the option on every compile line of
# REF_OFF, on none of REF and FORK, and at least one compile line each. Those
# counts are of the compile lines of ext/_hmmc.cpp only. FORK's second module,
# ext/_emissionsc.cpp, is counted apart: its setup.py gives it the option, so
# every one of its compile lines carries it, and there is at least one.
options() {
    echo "REF: $(option_count "$out/build_ref.log")"
    echo "REF_OFF: $(option_count "$out/build_ref_off.log")"
    echo "FORK: $(option_count "$out/build_fork.log")"
    echo "FORK ext/_emissionsc.cpp:" \
        "$(option_count "$out/build_fork.log" ext/_emissionsc.cpp)"
    for build in ref ref_off fork; do
        echo "-- $build"
        compile_lines "$out/build_$build.log"
    done
    echo "-- fork ext/_emissionsc.cpp"
    compile_lines "$out/build_fork.log" ext/_emissionsc.cpp
    for build in ref fork; do
        case "$(option_count "$out/build_$build.log")" in
            "0 of 0" | [1-9]*) return 1 ;;
        esac
    done
    count=$(option_count "$out/build_ref_off.log")
    if [ "${count%% of *}" = 0 ] \
        || [ "${count%% of *}" != "${count##* of }" ]; then
        return 1
    fi
    count=$(option_count "$out/build_fork.log" ext/_emissionsc.cpp)
    if [ "${count%% of *}" = 0 ] \
        || [ "${count%% of *}" != "${count##* of }" ]; then
        return 1
    fi
}

machine() {
    uname -sm
    compiler=$(compile_lines "$out/build_ref.log" | head -n 1 \
        | awk '{print $1}')
    "${compiler:-c++}" --version
    "$work/fork/bin/python" -I -c "import numpy; numpy.show_runtime()"
}

# dump ENV PACKAGE OUT [PICKLES]
dump() {
    if [ $# -eq 4 ]; then
        "$work/$1/bin/python" -I "$check" dump --package "$2" \
            --out "$out/$3" --load-pickles "$out/$4"
    else
        "$work/$1/bin/python" -I "$check" dump --package "$2" \
            --out "$out/$3"
    fi
}

# compare REF FORK ALLOW
compare() {
    "$work/fork/bin/python" -I "$check" compare "$out/$1" "$out/$2" \
        --allow "$3"
}

kernels() {
    "$work/fork/bin/python" -I -c "
import json
import sys

for path in sys.argv[1:]:
    name = path.rsplit('/', 1)[-1]
    try:
        with open(path + '/manifest.json') as file:
            env = json.load(file)['environment']
    except OSError:
        print(name, 'no dump')
        continue
    print(name, env['package'], env['package_version'],
          env['extension_sha256'])
" "$out/dump_ref" "$out/dump_ref_off" "$out/dump_fork_ref"
}

result() {
    tail -n 1 "$out/$1.log" 2>/dev/null
}

summary() {
    echo "Steps without a result: ${failed:-none}"
    echo "cell $cell on $(uname -sm); $(tr '\n' ' ' < "$out/pins.txt")"
    echo
    echo "$option on the compile lines of ext/_hmmc.cpp, no other file"
    echo "(expected: every line for REF_OFF, none for REF and FORK), and on"
    echo "those of FORK's ext/_emissionsc.cpp (expected: every line); the"
    echo "lines are in options.log"
    sed -n '1,4s/^/  /p' "$out/options.log"
    echo
    echo "The kernel file of each build, sha256:"
    sed 's/^/  /' "$out/kernels.log"
    ref_hash=$(awk '$1 == "dump_ref" {print $4}' "$out/kernels.log")
    ref_off_hash=$(awk '$1 == "dump_ref_off" {print $4}' "$out/kernels.log")
    if [ -z "$ref_hash" ] || [ -z "$ref_off_hash" ]; then
        echo "REF or REF_OFF has no kernel hash: see kernels.log."
    elif [ "$ref_hash" = "$ref_off_hash" ]; then
        echo "REF and REF_OFF are one file: the option changed no code here,"
        echo "and A is equal by construction."
    else
        echo "REF and REF_OFF differ: the option changed the compiled code"
        echo "here; A says whether that moves a result of the check."
    fi
    echo
    echo "Sums of products in the kernel (probe_contraction.py --package);"
    echo "wheel_ref and wheel_fork are the published wheels of hmmlearn $tag"
    echo "and of this package $published, files and sha256 in wheels.log:"
    for build in ref ref_off fork wheel_ref wheel_fork; do
        grep -E "$site_lines" "$out/probe_kernel_$build.log" \
            | sed "s/^/  $build: /"
    done
    grep -E '\.whl [0-9a-f]{64}$' "$out/wheels.log" | sed 's/^/  /'
    echo
    echo "Comparisons (0: equal; 1: differences, listed in the log;"
    echo "2: the dumps cannot be compared)"
    echo "  A  REF against REF_OFF, no exception: $(result compare_a)"
    echo "  B  REF against FORK, $allow: $(result compare_b)"
    echo "  C  REF_OFF against FORK, $allow: $(result compare_c)"
    echo
    echo "A 0: the option changes none of hmmlearn 0.3.3's results in this"
    echo "check on this machine, and C is expected to be 0 as well. A 1: it"
    echo "changes them; C then differs as well, unless every difference is"
    echo "in an array the gate's exceptions cover, and B is still expected to"
    echo "be 0. B 1: this package differs from hmmlearn 0.3.3 for a reason"
    echo "other than the option."
    echo
    echo "numpy (probe_contraction.py): probe_numpy.log"
    grep -E '^(einsum|feature sums): ' "$out/probe_numpy.log" | sed 's/^/  /'
}

step clone clone
step environments environments
step wheels wheels
step build_ref build_reference ref "$cppflags"
step build_ref_off build_reference ref_off "$cppflags $option"
step build_fork build_fork
step options options
step machine machine
step probe_numpy "$work/fork/bin/python" -I "$probe"
step probe_kernel_ref probe_kernel ref hmmlearn
step probe_kernel_ref_off probe_kernel ref_off hmmlearn
step probe_kernel_fork probe_kernel fork hmmcraft
step probe_kernel_wheel_ref probe_kernel wheels hmmlearn
step probe_kernel_wheel_fork probe_kernel wheels hmmcraft
step dump_ref dump ref hmmlearn dump_ref
step dump_ref_off dump ref_off hmmlearn dump_ref_off
step dump_fork_ref dump fork hmmcraft dump_fork_ref dump_ref
step dump_fork_ref_off dump fork hmmcraft dump_fork_ref_off dump_ref_off
step compare_a compare dump_ref dump_ref_off ""
step compare_b compare dump_ref dump_fork_ref "$allow"
step compare_c compare dump_ref_off dump_fork_ref_off "$allow"
step kernels kernels

# A comparison that finds differences (status 1) is a result.
failed=$(awk '!($2 == 0 || ($1 ~ /^compare_/ && $2 == 1)) {printf "%s ", $1}' \
    "$out/status.txt")
echo "== summary"
summary > "$out/summary.txt" 2>&1
cat "$out/summary.txt"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
        echo '```'
        cat "$out/summary.txt"
        echo '```'
    } >> "$GITHUB_STEP_SUMMARY"
fi
[ -z "$failed" ]
