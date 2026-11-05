#!/bin/bash
# Generate every radio map on elgin's GPU 2. Finished maps are skipped on rerun.
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=$HOME/miniconda3/envs/indoor-loc/bin/python
OPTS="--samples 2e8 --depth 8 --batch 1"

run() {
    if [ -f "data/radio_maps/$1__$2.npz" ]; then
        echo "SKIP $1 $2"
        return
    fi
    PYTHONPATH=. $PY scripts/generate_radio_maps.py --set "$1" --band "$2" $OPTS | tail -1
}

run wifi wifi_2g4
run wifi wifi_5g
run A_even_ceiling_10m ble
run B_even_ceiling_15m ble
run C_even_low_15m ble
run D_uneven ble
echo ALL_DONE
