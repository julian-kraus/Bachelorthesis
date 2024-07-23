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
import tensorflow as tf
def eval_training(history, metrics):
     history_dict = history.history
     epochs = range(1, len(history_dict['loss']) + 1)

     # Metrics to plot
     metrics_names = [metric.name for metric in metrics]

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
def eval(model, test_padded, test_labels, label_encoder):
     # Predict classes on the test data
     test_predictions = model.predict(test_padded)
     test_predicted_classes = np.argmax(test_predictions, axis=1)
     test_true_classes = np.argmax(test_labels, axis=1)

     # Decode integer labels to original labels
     test_predicted_labels = label_encoder.inverse_transform(test_predicted_classes)
     test_true_labels = label_encoder.inverse_transform(test_true_classes)

     # Confusion matrix
     conf_matrix = confusion_matrix(test_true_labels, test_predicted_labels, labels=label_encoder.classes_)
     plt.figure(figsize=(6, 3))
     sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=label_encoder.classes_,
                 yticklabels=label_encoder.classes_)
     plt.xlabel('Predicted Label')
     plt.ylabel('True Label')
     plt.title('Confusion Matrix')
     plt.show()

     from sklearn.metrics import f1_score
     print("F1: " + f1_score(test_true_labels, test_predicted_labels, average='micro'))


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
     class_weights = None
     sample_weights = None
     if class_weights or sample_weights:
          class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_enc), y=train_labels_enc)
          class_weights = {np.unique(train_labels_enc)[i]: w for i, w in enumerate(class_weights)}
          print(class_weights)
          if sample_weights:
               sample_weights = np.array([class_weights[label] for label in train_labels_enc])
          # Count each class
          values, counts = np.unique(df[predict_label], return_counts=True)
          class_distribution = dict(zip(values, counts))
          print("Original Class Distribution:", class_distribution)

          # Encode the keys of class_distribution
          encoded_keys = label_encoder.transform(list(class_distribution.keys()))
          encoded_class_distribution = dict(zip(encoded_keys, counts))
          print("Encoded Class Distribution:", encoded_class_distribution)
          if not class_weights:
              class_weights = None
     if not one_hot:
          train_labels = train_labels_enc
          valid_labels = valid_labels_enc
          test_labels = test_labels_enc
     return train_padded, valid_padded, test_padded, train_labels, valid_labels, test_labels, label_encoder, num_classes, max_length, word_index, class_weights, sample_weights
