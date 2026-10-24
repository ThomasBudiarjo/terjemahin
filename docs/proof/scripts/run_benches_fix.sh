#!/bin/bash
cd /home/user/terjemahin/apps/worker
B=".venv/bin/python -m terjemahin_worker.eval bench"
O=out/final-fixer/bench
mkdir -p $O; : > $O/done.txt
run() { name=$1; page=$2; gt=$3; shift 3; S=$(date +%s); $B --page $page --gt $gt --out $O/$name --no-warm "$@" > $O/$name.log 2>&1; echo "$name exit $? $(( $(date +%s)-S ))s" >> $O/done.txt; }
run psi5 eval/fixtures/real/psimera_val_000005.jpg eval/fixtures/real/psimera_val_000005.gt.json
run psi2 eval/fixtures/real/psimera_val_000002.jpg eval/fixtures/real/psimera_val_000002.gt.json
run arms out/judge1/m109/ARMS_015.jpg eval/heldout/ARMS_015.gt.json
run arisa out/judge1/m109/Arisa_010.jpg eval/heldout/Arisa_010.gt.json
run p75q75 out/judge3/p75q75.jpg eval/fixtures/page01.gt.json
run page01_q40 out/proof/hostile/page01_lowres_q40.jpg eval/fixtures/page01.gt.json
run synth_000002 out/proof/synth/gen/det/images/train/000002.jpg out/proof/synth/gt/000002.gt.json
run synth_000003 out/proof/synth/gen/det/images/train/000003.jpg out/proof/synth/gt/000003.gt.json
echo ALLDONE >> $O/done.txt
S=$(date +%s); $B --page eval/fixtures/page01.png --gt eval/fixtures/page01.gt.json --out out/final-fixer/bench_matrix --matrix > out/final-fixer/bench_matrix.log 2>&1; echo "matrix exit $? $(( $(date +%s)-S ))s" >> $O/done.txt
.venv/bin/python -m terjemahin_worker.eval lines -t nllb -t m2m100 -t chain --out out/final-fixer/lines_final > out/final-fixer/lines_final.log 2>&1; echo "lines exit $?" >> $O/done.txt
echo FINISHED >> $O/done.txt
