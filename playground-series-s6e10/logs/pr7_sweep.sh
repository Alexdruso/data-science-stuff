#!/bin/bash
# PR-7 stage-1 sweep: fold 1 only, two workers sharing the GPU.
cd /home/alexdruso/Desktop/data-science-stuff/playground-series-s6e10
source ../.venv/bin/activate
export PYTHONUNBUFFERED=1
run() { python src/train_realmlp.py --fold1 "$@" 2>&1 | grep -E "FOLD1|best score|Error|Traceback"; }
worker_a() {
  run --seed 43                                        # noise control (v1 config)
  run --tag e6  --override '{"epochs": 6}'
  run --tag e8  --override '{"epochs": 8}'
  run --tag e10 --override '{"epochs": 10}'
}
worker_b() {
  run --tag e8w256  --override '{"epochs": 8, "hidden_dims": [256, 256, 256]}'
  run --tag e8w1024 --override '{"epochs": 8, "hidden_dims": [1024, 1024, 1024]}'
  run --tag e8lr005 --override '{"epochs": 8, "lr": 0.005}'
  run --tag e8lr02  --override '{"epochs": 8, "lr": 0.02}'
  run --tag e8reg   --override '{"epochs": 8, "dropout": 0.15, "weight_decay": 0.05}'
}
worker_a > logs/pr7_a.log &
worker_b > logs/pr7_b.log &
wait
echo SWEEP_DONE
