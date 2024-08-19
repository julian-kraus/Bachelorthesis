#!/bin/bash

echo "Script started"

# Define the path to the notebooks
notebook_path="./LLM"

# List of notebooks to run
notebooks=(
    #"$notebook_path/implementation_attention.ipynb"
    "$notebook_path/classification/BERT_training_balanced_oversampled_med_epoch_15.ipynb"
    "$notebook_path/classification/BERT_pooling_max.ipynb"
)


# Execute each notebook
for notebook in "${notebooks[@]}"; do
    echo "About to run: $notebook"
    jupyter nbconvert --to notebook --execute --inplace "$notebook" || echo "Failed to execute $notebook"
    echo "$notebook completed."
done

echo "All notebooks have been executed."
