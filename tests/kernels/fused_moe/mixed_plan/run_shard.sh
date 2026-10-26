#!/bin/bash
# Job #1 shards (probe 01). usage: run_shard.sh <i>
cd /root/kp
export PYTHONPATH=/root/kp/src:/root/kp:$PYTHONPATH
D=tests/kernels/fused_moe/mixed_plan
O=$CDK_OUTPUT_DIR/kplan
mkdir -p $O
python3 -c "import vllm_torchtpu.kernels.fused_moe.v2.host as h; print('host from', h.__file__)"
md5sum src/vllm_torchtpu/kernels/fused_moe/v2/*.py $D/*.py
A1ARMS=plan,a1,a1_nohi,a1_rm_s3,a1_rm_ydot
A2R='combine,a2_inline,a2_inline@nodma,a2_inline@nocompute,a2_inline@nopick,a2_inline@noout,a2_inline@rowconst,a2_inline@nodma;nocompute;nopick;noout'
A22='combine,a2_inline,a2_ring,a2m,a2_inline@noout,a2_inline@rowconst,a2_inline@nodma;nocompute;nopick;noout,a2m@nodma,a2m@nocompute,a2m@nopick,a2m@noout,a2m@rowconst,a2m@nodma;nocompute;noout'
A23='combine,a2_inline,a2s,a2_ring,a2s_ring,a2s@noout,a2s@nodma,a2s@nocompute,a2s@nopick,a2s@nodma;nocompute;nopick;noout'
case $1 in
0) timeout -s KILL 2400 python3 $D/bench.py --cells mix128,mix256 --kinds served,served_pad --arms plan,combine --trace --out $O/b0 ;;
1) pip install -q pytest >/dev/null 2>&1; timeout -s KILL 2400 python3 -m pytest -q -s -p no:cacheprovider $D/test_accuracy.py ;;
2) timeout -s KILL 1800 python3 $D/probe_primitives.py ;;
3) timeout -s KILL 2400 python3 $D/bench.py --cells mix32,mix64,mix128,mix256 --kinds uniform,one_shard --arms plan --me 0 --out $O/b3 ;;
4) KPLAN_RS=16,96,256 KPLAN_PROBE_OUT=kplan_probe00b.json timeout -s KILL 1800 python3 $D/probe_primitives.py ;;
5) timeout -s KILL 1800 python3 $D/probe_lowering.py ;;
6) timeout -s KILL 1500 python3 $D/probe_tripcount.py ;;
# probe 04 (stage A1)
10) timeout -s KILL 2400 python3 $D/bench.py --cells mix128 --kinds served,served_pad --arms $A1ARMS --rs 16,96,256 --trace --out $O/p4_128 ;;
11) timeout -s KILL 2400 python3 $D/bench.py --cells mix256 --kinds served,served_pad --arms $A1ARMS --rs 16,96,256 --trace --out $O/p4_256 ;;
12) timeout -s KILL 2400 python3 $D/bench.py --cells mix32,mix64,mix128,mix256 --kinds one_shard,uniform --arms plan,a1,a1_rm_s3 --me 0 --rs 16,96,256 --out $O/p4_empty ;;
# probe 05 (fixed-cost attribution, stage removal arms)
20) timeout -s KILL 2400 python3 $D/bench.py --cells mix128 --kinds served --arms plan,a1,a1@skip:s1,a1@skip:s2,a1@skip:s3,a1@skip:s1\;s2\;s3,a1@rm_s3 --rs 16,96,256 --out $O/p5_128 ;;
21) timeout -s KILL 2400 python3 $D/bench.py --cells mix256 --kinds served --arms plan,a1,a1@skip:s1,a1@skip:s2,a1@skip:s3,a1@skip:s1\;s2\;s3,a1@rm_s3 --rs 16,96,256 --out $O/p5_256 ;;
22) timeout -s KILL 2400 python3 $D/bench.py --cells mix32 --kinds uniform --arms plan,a1,a1@skip:s1,a1@skip:s2,a1@skip:s3,a1@skip:s1\;s2\;s3,a1@rm_s3 --rs 16,96,256 --out $O/p5_32 ;;
23) timeout -s KILL 2400 python3 $D/bench.py --cells mix128 --kinds served --arms plan,a1,a1@skip:s1,a1@skip:s2,a1@skip:s3,a1@skip:s1\;s2\;s3,a1@rm_s3 --rs 16,96,256 --trace --out $O/p5_128b ;;
# stage A2 (mixed combine): probe 2x-a2-first
30) pip install -q pytest >/dev/null 2>&1; timeout -s KILL 2400 python3 -m pytest -q -s -p no:cacheprovider $D/test_accuracy.py -k a2 ;;
31) timeout -s KILL 2400 python3 $D/bench.py --cells mix128 --kinds served --arms combine,pk128,a2,a2_inline,a2_ring,a2@nodma,a2@nocompute,a2@nopick --rs 16,96,256 --trace --out $O/a2_128 ;;
32) timeout -s KILL 2400 python3 $D/bench.py --cells mix256 --kinds served --arms combine,a2,a2_inline,a2_ring,a2@nodma,a2@nocompute,a2@nopick --rs 16,96,256 --trace --out $O/a2_256 ;;
33) timeout -s KILL 2400 python3 $D/bench.py --cells tw16,tw4 --kinds served --arms combine,a2,a2_ring --rs 16,96,256 --trace --out $O/a2_tw ;;
34) timeout -s KILL 2400 python3 $D/bench.py --cells mix128 --kinds served,served_pad --arms combine,a2,a2_inline,a2_ring,a2@nodma,a2@nocompute,a2@nopick --rs 16,96,256 --out $O/a2_128r ;;
35) timeout -s KILL 2400 python3 $D/bench.py --cells mix256 --kinds served,served_pad --arms combine,a2,a2_inline,a2_ring,a2@nodma,a2@nocompute,a2@nopick --rs 16,96,256 --out $O/a2_256r ;;
36) pip install -q pytest >/dev/null 2>&1; timeout -s KILL 2400 python3 -m pytest -q -s -p no:cacheprovider -m multichip tests/kernels/fused_moe/test_fused_ep_moe_v2_mixed_combine.py ;;
# probe 21 (2x-a2-attr): attribution of a2_inline's kernel time
40) pip install -q pytest >/dev/null 2>&1; timeout -s KILL 2400 python3 -m pytest -q -s -p no:cacheprovider $D/test_accuracy.py -k a2 ;;
41) timeout -s KILL 2400 python3 $D/bench.py --cells mix128 --kinds served --arms "$A2R" --rs 16,96,256 --trace --out $O/p21_128 ;;
42) timeout -s KILL 2400 python3 $D/bench.py --cells mix256 --kinds served --arms "$A2R" --rs 16,96,256 --trace --out $O/p21_256 ;;
43) timeout -s KILL 2400 python3 $D/bench.py --cells mix128 --kinds served_pad --arms "$A2R" --rs 16,96,256 --trace --out $O/p21_128p ;;
# probe 22 (2x-a2-mxu): a2m (MXU pick) + attribution; supersedes probe 21's job (deleted unstarted)
50) pip install -q pytest >/dev/null 2>&1; timeout -s KILL 2400 python3 -m pytest -q -s -p no:cacheprovider $D/test_accuracy.py -k a2 ;;
51) timeout -s KILL 2700 python3 $D/bench.py --cells mix128 --kinds served --arms "$A22" --rs 16,96,256 --trace --out $O/p22_128 ;;
52) timeout -s KILL 2700 python3 $D/bench.py --cells mix256 --kinds served --arms "$A22" --rs 16,96,256 --trace --out $O/p22_256 ;;
53) timeout -s KILL 2700 python3 $D/bench.py --cells tw16,tw4,mix128 --kinds served,served_pad --arms combine,a2_inline,a2_ring,a2m --rs 16,96,256 --trace --out $O/p22_tw ;;
# probe 23 (2x-a2-strided): strided relayout (a2s, a2s_ring) + attribution of a2s
60) pip install -q pytest >/dev/null 2>&1; timeout -s KILL 2400 python3 -m pytest -q -s -p no:cacheprovider $D/test_accuracy.py -k "a2s or a2_inline" ;;
61) timeout -s KILL 2700 python3 $D/bench.py --cells mix128 --kinds served --arms "$A23" --rs 16,96,256 --trace --out $O/p23_128 ;;
62) timeout -s KILL 2700 python3 $D/bench.py --cells mix256 --kinds served --arms "$A23" --rs 16,96,256 --trace --out $O/p23_256 ;;
63) timeout -s KILL 2700 python3 $D/bench.py --cells tw16,tw4,mix128,mix256 --kinds served_pad --arms combine,a2_inline,a2s,a2s_ring --rs 16,96,256 --trace --out $O/p23_tw ;;
esac
