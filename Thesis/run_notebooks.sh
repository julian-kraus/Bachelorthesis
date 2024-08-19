#!/bin/bash

echo "Script started"

# Define the path to the notebooks
notebook_path="./LSTM"

# List of notebooks to run
notebooks=(
    "$notebook_path/implementation_bert.ipynb"
    "$notebook_path/implementation_tuner.ipynb"
    "$notebook_path/implementation_word2vec.ipynb"
    "$notebook_path/implementation_attention.ipynb"
    #"$notebook_path/implementation_baseline.ipynb"
    "$notebook_path/implementation_batchnormalization.ipynb"
    "$notebook_path/implementation_conv_layer.ipynb"
    "$notebook_path/implementation_dropout_less.ipynb"
    "$notebook_path/implementation_dropout_more.ipynb"
    #"$notebook_path/implementation_dropout_recurrent.ipynb"
    "$notebook_path/implementation_L1_L2.ipynb"
    "$notebook_path/implementation_L1.ipynb"
    "$notebook_path/implementation_L2.ipynb"
    "$notebook_path/implementation_layers_less.ipynb"
    "$notebook_path/implementation_layers_more.ipynb"
    #"$notebook_path/implementation_learning_rate_finder.ipynb"
    "$notebook_path/implementation_learning_rate_scheduler.ipynb"
    "$notebook_path/implementation_oversampling.ipynb"
    "$notebook_path/implementation_undersampling.ipynb"
    "$notebook_path/implementation_weights.ipynb"
)


# Execute each notebook
for notebook in "${notebooks[@]}"; do
    echo "About to run: $notebook"
    jupyter nbconvert --to notebook --execute --inplace "$notebook" || echo "Failed to execute $notebook"
    echo "$notebook completed."
done

echo "All notebooks have been executed."
