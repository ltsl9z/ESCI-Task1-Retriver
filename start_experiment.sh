#!/bin/bash

# declare locale parameters
LOCALES="us"
CONFIG_NAME="model_config.yml"
RESULT_DIR="evaluation_results"

echo "----------------------------- start fine tuning pipeline -----------------------------."
python fine_tune.py \
    --config_name=$CONFIG_NAME \
    --locale=$LOCALES
echo "----------------------------- end fine tuning pipeline -----------------------------."

echo "----------------------------- start model evaluation -----------------------------."
python evaluation.py \
    --config_name=$CONFIG_NAME \
    --result_dir=$RESULT_DIR \
    --locale=$LOCALES \
echo "----------------------------- end model evaluation -----------------------------."