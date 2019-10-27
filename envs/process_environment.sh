#!/bin/bash
# Process environment file to remove OS-specific packages

if [[ "$OSTYPE" == "darwin"* ]]; then
# Remove libcxx, libcxxabi, libgfortran, appnope from environment file
    sed -i.bak '/.*libcxx.*/d;/.*libgfortran.*/d;/.*appnope.*/d' ~/capstone/envs/environment.yaml && rm ~/capstone/envs/environment.yaml.bak
fi
