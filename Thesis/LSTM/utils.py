from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import compute_class_weight
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, Conv1D, GlobalMaxPooling1D, Concatenate, AdditiveAttention
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import AdditiveAttention
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='Precision is ill-defined and being set to 0.0 in labels with no predicted samples.')
from absl import logging
logging.set_verbosity(logging.ERROR)  # Ignore INFO and WARNING
model_type = "lstm"

def get_standard_parameters():
    optimizer = Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.F1Score(average="macro", name='f1_score'),
    ]
    early_stopping_callback = EarlyStopping(
        monitor='val_f1_score',  
        patience=3,         
        restore_best_weights=True,
        verbose=1
    )
    embedding_dim = 300
    lstm_units = 64
    epochs = 15
    batch_size = 32 
    return optimizer, loss, metrics, early_stopping_callback, embedding_dim, lstm_units, epochs, batch_size


def eval_training(history, metrics):
     history_dict = history.history
     epochs = range(1, len(history_dict['loss']) + 1)

     # Metrics to plot
     metrics_names = [metric.name for metric in metrics]
     metrics_names = metrics_names + ['loss']

     plt.figure(figsize=(10, 7))

     for i, metric in enumerate(metrics_names, 1):
          plt.subplot(2, 3, i)
          plt.plot(epochs, history_dict[metric], 'bo', label=f'Training {metric}')
          plt.plot(epochs, history_dict[f'val_{metric}'], 'b', label=f'Validation {metric}')
          plt.title(f'Training and Validation {metric.capitalize()}')
          plt.xlabel('Epochs')
          plt.ylabel(metric.capitalize())
          plt.legend()

     plt.tight_layout()
     plt.show()

def eval(model, test_padded, test_labels, label_encoder, model_name=""):
     # Predict classes on the test data
     test_predictions = model.predict(test_padded)
     test_predicted_classes = np.argmax(test_predictions, axis=1)
     test_true_classes = np.argmax(test_labels, axis=1)

     # Decode integer labels to original labels
     test_predicted_labels = label_encoder.inverse_transform(test_predicted_classes)
     test_true_labels = label_encoder.inverse_transform(test_true_classes)

     class_report = classification_report(test_true_labels, test_predicted_labels, output_dict=True)
     class_report_df = pd.DataFrame(class_report).transpose()
     class_report_df = class_report_df.round(2)
     fig, ax = plt.subplots(figsize=(7, 3))  # Adjust size as needed

     # Hide axes
     ax.axis('tight')
     ax.axis('off')

     # Create the table and adjust properties as needed
     the_table = ax.table(cellText=class_report_df.values, colLabels=class_report_df.columns, rowLabels=class_report_df.index, loc='center', cellLoc = 'center', rowLoc = 'center')

     # Optionally, resize cells
     the_table.auto_set_font_size(False)
     the_table.set_fontsize(10)
     the_table.scale(1.2, 1.2)  # Scale table size
     plt.title(f"{model_name}", fontsize=12, weight='bold', pad=20)
     # Display the plot
     plt.show()
     # Save to CSV
     if model_name != "":
          class_report_df.to_csv("../reports/" + model_type + "/" + model_name + "_report")

def compare_models(models_data):
    for model_data in models_data:
        model, test_data, test_labels, label_encoder, model_name = model_data
        eval(model, test_data, test_labels, label_encoder, model_name)

def process_train_test_data(train_df, valid_df, test_df, data_label, predict_label, lables=None, class_weights=False, sample_weights=False, one_hot=True):
     train_texts = train_df[data_label]
     valid_texts = valid_df[data_label]
     test_texts = test_df[data_label]

     # Tokenize texts
     tokenizer = Tokenizer(num_words=10000)
     tokenizer.fit_on_texts(train_texts)  # Only fit on train data
     word_index = tokenizer.word_index
     train_sequences = tokenizer.texts_to_sequences(train_texts)
     valid_sequences = tokenizer.texts_to_sequences(valid_texts)
     test_sequences = tokenizer.texts_to_sequences(test_texts)
     max_length = max(len(sequence) for sequence in train_sequences) + 30
     # Padding sequences
     train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
     valid_padded = pad_sequences(valid_sequences, maxlen=max_length, padding='post')
     test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

     # Initialize the label encoder
     all_labels = pd.concat([train_df[predict_label], valid_df[predict_label], test_df[predict_label]])
     label_encoder = LabelEncoder()
     label_encoder.fit(all_labels)
     # Fit label encoder and return encoded labels as integers
     train_labels_enc = label_encoder.transform(train_df[predict_label])
     valid_labels_enc = label_encoder.transform(valid_df[predict_label])
     test_labels_enc = label_encoder.transform(test_df[predict_label])

     # Convert labels to categorical one-hot encoding
     train_labels = to_categorical(train_labels_enc)
     valid_labels = to_categorical(valid_labels_enc)
     test_labels = to_categorical(test_labels_enc)

     label_counts = train_df[predict_label].value_counts()
     num_classes = len(train_df[predict_label].unique())
     print(label_counts)
     # Plot the distribution of labels
     plt.figure(figsize=(3, 2))
     label_counts.plot(kind='bar')
     plt.title('Distribution of Labels')
     plt.xlabel('Labels')
     plt.ylabel('Frequency')
     plt.xticks(rotation=0)
     plt.show()
     class_weight = None
     sample_weight = None
     if class_weights or sample_weights:
          class_weight = compute_class_weight('balanced', classes=np.unique(train_labels_enc), y=train_labels_enc)
          class_weight = {np.unique(train_labels_enc)[i]: w for i, w in enumerate(class_weight)}
          print(class_weight)
          if sample_weight:
               sample_weight = np.array([class_weight[label] for label in train_labels_enc])
          # Count each class
          values, counts = np.unique(train_df[predict_label], return_counts=True)
          class_distribution = dict(zip(values, counts))
          print("Original Class Distribution:", class_distribution)

          # Encode the keys of class_distribution
          encoded_keys = label_encoder.transform(list(class_distribution.keys()))
          encoded_class_distribution = dict(zip(encoded_keys, counts))
          print("Encoded Class Distribution:", encoded_class_distribution)
          if not class_weight:
              class_weight = None
     if not one_hot:
          train_labels = train_labels_enc
          valid_labels = valid_labels_enc
          test_labels = test_labels_enc
     return train_padded, valid_padded, test_padded, train_labels, valid_labels, test_labels, label_encoder, num_classes, max_length, word_index, class_weight, sample_weight

def get_train_test_data(df, data_label, predict_label, balanced=False, lables=None, class_weights=False, sample_weights=False, one_hot=True):

     df = df.sample(frac=1).reset_index(drop=True)
     if lables is not None:
          df = df[df['label'].isin(lables)]

     if balanced:
          # Identify the class with the smallest number of samples
          min_class_count = df['label'].value_counts().min()

          # Separate majority and minority classes
          df = df.groupby('label').apply(lambda x: x.sample(min_class_count)).reset_index(drop=True)

     # Split data into training and temporary data (the remaining 40%)
     train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df[predict_label])

     # Split the temporary data into validation and test sets (each 50% of temporary, thus 20% of total each)
     valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[predict_label])

     # Check the size of each set
     print("Training set size:", len(train_df))
     print("Validation set size:", len(valid_df))
     print("Test set size:", len(test_df))
     # Prepare data for training
     return process_train_test_data(train_df, valid_df, test_df, data_label, predict_label, lables, class_weights, sample_weights, one_hot)

import os
import pickle
from keras.models import load_model

def save_for_evaluation(model, history, model_name, test_data, test_labels, label_encoder):
    # Create model and data directories if they don't exist
    model_dir = f'../models/{model_type}/{model_name}'
    data_dir = f'../data/{model_type}/{model_name}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save the model
    model_path = f'{model_dir}/{model_name}.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save the history
    history_path = f'{data_dir}/{model_name}_history.pkl'
    with open(history_path, 'wb') as file:
        pickle.dump(history, file)

    # Save the test data
    data_path = f'{data_dir}/{model_name}_test_data.pkl'
    with open(data_path, 'wb') as file:
        pickle.dump(test_data, file)

    # Save the test labels
    label_path = f'{data_dir}/{model_name}_test_label.pkl'
    with open(label_path, 'wb') as file:
        pickle.dump(test_labels, file)

    # Save the encoder
    encoder_path = f'{data_dir}/{model_name}_encoder.pkl'
    with open(encoder_path, 'wb') as file:
        pickle.dump(label_encoder, file)

    print("Data saved")


def load_for_evaluation(model_name):
    # Define paths for the saved files
    model_path = f'../models/{model_type}/{model_name}/{model_name}.h5'
    history_path = f'../data/{model_type}/{model_name}/{model_name}_history.pkl'
    data_path = f'../data/{model_type}/{model_name}/{model_name}_test_data.pkl'
    label_path = f'../data/{model_type}/{model_name}/{model_name}_test_label.pkl'
    encoder_path = f'../data/{model_type}/{model_name}/{model_name}_encoder.pkl'
    
    # Load the model
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Load the history
    with open(history_path, 'rb') as file:
        history = pickle.load(file)
    
    # Load the test data
    with open(data_path, 'rb') as file:
        test_data = pickle.load(file)

    # Load the test labels
    with open(label_path, 'rb') as file:
        test_labels = pickle.load(file)

    # Load the encoder
    with open(encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)
    
    print("Data loaded successfully")
    return model, test_data, test_labels, label_encoder, model_name
