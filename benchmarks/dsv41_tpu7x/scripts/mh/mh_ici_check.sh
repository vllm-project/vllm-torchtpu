#!/usr/bin/env bash
# Is the 2-host decode unit using ICI or DCN for its collectives?  Run from the driver once the
# DP16 head is healthy. Evidence collected:
#   1. GKE slice facts: node labels (topology, num-hosts) and the TPU env GKE injects into pods.
#   2. Runtime facts: torch_tpu / libtpu log lines about topology / slice builder / megascale.
#   3. Data-path test: host NIC byte counters sampled idle vs. under a c=512 decode load.
#      EP all-to-all at 192-384 tok/rank/step is O(100 MB)/rank/step; over DCN that is >= GB/s on
#      the NICs, over ICI it is ~0 on the NICs.
set -u
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps
HERE=$(cd "$(dirname "$0")" && pwd); OUT=$HERE/../../runs/mh/ici_check_$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
k() { kubectl --context "$CTX" "$@"; }
P0=$(k get pods -l dsv41-thread=mh -o jsonpath='{range .items[*]}{.metadata.name} {.metadata.annotations.batch\.kubernetes\.io/job-completion-index}{"\n"}{end}' | awk '$2==0{print $1}')
P1=$(k get pods -l dsv41-thread=mh -o jsonpath='{range .items[*]}{.metadata.name} {.metadata.annotations.batch\.kubernetes\.io/job-completion-index}{"\n"}{end}' | awk '$2==1{print $1}')
[ -n "$P0" ] || { echo "no mh pods"; exit 1; }
{
echo "### 1. GKE slice facts"
k get nodes -l cloud.google.com/gke-nodepool=cpk-spot-2x2x2 -o custom-columns='NAME:.metadata.name,TOPO:.metadata.labels.cloud\.google\.com/gke-tpu-topology,HOSTS:.metadata.labels.cloud\.google\.com/gke-tpu-num-hosts,ACC:.metadata.labels.cloud\.google\.com/gke-tpu-accelerator,ZONE:.metadata.labels.topology\.kubernetes\.io/zone'
for P in $P0 $P1; do
  echo "--- $P (pod-injected TPU env, before our launcher blanks it)"
  k exec "$P" -- bash -c 'env | grep -E "^TPU_|^MEGASCALE|^TORCH_TPU" | sort; echo "accel devices:"; ls /dev/accel* /dev/vfio 2>/dev/null | tr "\n" " "; echo; cat /sys/class/accel/accel0/device/numa_node 2>/dev/null'
done
echo; echo "### 2. Runtime topology lines (head)"
k exec "$P0" -- bash -c 'grep -iE "TORCH_TPU_TOPOLOGY=|slicebuilder|multihost DP|megascale|ici|mesh shape|device_count|num_devices|host_bounds|chips_per_host" /dev/shm/mh_decode.log | grep -v env_vars.h | sort -u | head -40 | cut -c1-300'
echo; echo "### 3. NIC counters idle vs load (host0 and host1), bytes/s"
sample() { k exec "$1" -- bash -c 'for n in /sys/class/net/*; do d=$(basename $n); [ "$d" = lo ] && continue; echo "$d $(cat $n/statistics/rx_bytes) $(cat $n/statistics/tx_bytes)"; done'; }
echo "-- idle (10 s)"; A0=$(sample $P0); B0=$(sample $P1); sleep 10; A1=$(sample $P0); B1=$(sample $P1)
python3 - "$A0" "$A1" "$B0" "$B1" 10 <<'EOF'
import sys
def parse(s): return {l.split()[0]:(int(l.split()[1]),int(l.split()[2])) for l in s.strip().splitlines() if len(l.split())==3}
for name,(a,b) in {"host0":(sys.argv[1],sys.argv[2]),"host1":(sys.argv[3],sys.argv[4])}.items():
    x,y=parse(a),parse(b); dt=float(sys.argv[5])
    print(name, {k:(round((y[k][0]-x[k][0])/dt/1e6,1), round((y[k][1]-x[k][1])/dt/1e6,1)) for k in y if k in x}, "MB/s (rx,tx)")
EOF
echo "-- under load: c=512 (32 seqs/rank) ctx 2K, sampling 20 s in steady state"
k cp "$HERE/../pd/x1_step_anatomy.py" "$P0:/tmp/x1_step_anatomy.py" >/dev/null 2>&1
k exec "$P0" -- bash -c 'nohup setsid python3 -u /tmp/x1_step_anatomy.py --decode http://127.0.0.1:9400 --ctx 2000 --out-len 1024 --concs 512 --dp 16 --k 5 --out /tmp/x1_icicheck.json > /tmp/x1_icicheck.log 2>&1 < /dev/null & disown'
sleep 45
A0=$(sample $P0); B0=$(sample $P1); sleep 20; A1=$(sample $P0); B1=$(sample $P1)
python3 - "$A0" "$A1" "$B0" "$B1" 20 <<'EOF'
import sys
def parse(s): return {l.split()[0]:(int(l.split()[1]),int(l.split()[2])) for l in s.strip().splitlines() if len(l.split())==3}
for name,(a,b) in {"host0":(sys.argv[1],sys.argv[2]),"host1":(sys.argv[3],sys.argv[4])}.items():
    x,y=parse(a),parse(b); dt=float(sys.argv[5])
    print(name, {k:(round((y[k][0]-x[k][0])/dt/1e6,1), round((y[k][1]-x[k][1])/dt/1e6,1)) for k in y if k in x}, "MB/s (rx,tx)")
EOF
echo "(for reference: EP a2a at 192 tok/rank/step over DCN would be >= 1,000 MB/s per host; <= tens of MB/s means ICI)"
sleep 60; k exec "$P0" -- bash -c 'grep "\[X1\]" /tmp/x1_icicheck.log | cut -c1-200'
} 2>&1 | tee "$OUT/ici_check.txt"
echo "saved $OUT/ici_check.txt"
