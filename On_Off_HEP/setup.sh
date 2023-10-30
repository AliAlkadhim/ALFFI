#!/bin/bash
# conda activate ALFFI
export ALFFI_ON_OFF=`pwd`
export ALFFI_ON_OFF_DATA=$ALFFI_ON_OFF/data
export ALFFI_ON_OFF_SRC=$ALFFI_ON_OFF/src
export ALFFI_ON_OFF_MODELS=$ALFFI_ON_OFF/models
export ALFFI_ON_OFF_IMAGES=$ALFFI_ON_OFF/images

# export deps with conda env create -n ALFFI python=3.9 scipy numpy pandas torch numba matplotlib joblib importlib