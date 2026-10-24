#!/bin/bash
cd /home/user/terjemahin/apps/worker
B=".venv/bin/python -m terjemahin_worker.eval bench"
run() { name=$1; page=$2; gt=$3; S=$(date +%s); $B --page $page --gt $gt --out out/proof/extra/$name > out/proof/extra/$name.log 2>&1; echo "$name exit $? $(( $(date +%s)-S ))s" >> out/proof/extra/done.txt; }
mkdir -p out/proof/extra; : > out/proof/extra/done.txt
run synth_000002 out/proof/synth/gen/det/images/train/000002.jpg out/proof/synth/gt/000002.gt.json
run synth_000003 out/proof/synth/gen/det/images/train/000003.jpg out/proof/synth/gt/000003.gt.json
run synth_000007 out/proof/synth/gen/det/images/train/000007.jpg out/proof/synth/gt/000007.gt.json
run synthh_000000 out/proof/synth/genh/det/images/train/000000.jpg out/proof/synth/gth/000000.gt.json
run tall_strip out/proof/hostile/tall_strip.png out/proof/hostile/tall_strip.gt.json
run page01_lowres out/proof/hostile/page01_lowres_q40.jpg eval/fixtures/page01.gt.json
echo ALLDONE >> out/proof/extra/done.txt
