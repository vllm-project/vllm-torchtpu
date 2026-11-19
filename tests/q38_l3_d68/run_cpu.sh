#!/bin/bash
# q38 L3 D68 CPU proof body (CDK parasim-cpu job; run from the unpacked tree root). Exit code = verdict.
# Legs: P0 versions + src hash; P1 new D68 tests; P2 existing AR-FIX math + wiring tests (flag-off regression);
# P3 H8 kernel tests (unchanged kernel, sanity); P4 D68 source mutants (all must be killed).
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH="$PWD/src"
fail=0
echo "P0 $(date -u +%FT%TZ) python $(python3 -c 'import sys;print(sys.version.split()[0])') jax $(python3 -c 'import jax;print(jax.__version__)' 2>&1 | tail -1)"
echo "P0 src $(find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10) commit $(cat COMMIT 2>/dev/null)"
leg() { local n=$1; shift; echo "== $n: $*"; "$@" > /tmp/leg_$n.log 2>&1; local rc=$?; tail -4 /tmp/leg_$n.log; echo "$n rc=$rc"; [ $rc = 0 ] || fail=1; }
leg P1 python3 -m pytest -q -p no:cacheprovider --noconftest tests/q38_l3_d68/test_tp_ar_pad16.py
leg P2 python3 -m pytest -q -p no:cacheprovider --noconftest tests/arfix/test_tp_ar_decode_math.py tests/arfix/test_tp_ar_decode_wiring.py
leg P3 python3 -m pytest -q -p no:cacheprovider --noconftest tests/arfix/test_tp8_hier_all_reduce.py
leg P4 python3 tests/q38_l3_d68/q38_l3_d68_mutrun.py
cat /tmp/leg_P4.log
echo "Q38_L3_D68_SUMMARY $( [ $fail = 0 ] && echo PASS || echo FAIL )"
exit $fail
