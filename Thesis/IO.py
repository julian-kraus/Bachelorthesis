import pandas as pd
import pickle
from keras.models import load_model as keras_load_model
import os


def load_excel(file_path):
    # Using 'ISO-8859-1' encoding which should handle German characters
    df = pd.read_excel(file_path)
    # Assuming the data of interest is in the third column (index 2)
    return df

def load_csv(file_path):
    # Using 'ISO-8859-1' encoding which should handle German characters
    df = pd.read_csv(file_path)
    # Assuming the data of interest is in the third column (index 2)
    return df

def save_model(model, history, type, name):
    path = './models/' + type + '/'
    with open(path + name + '_history.pkl', 'wb') as file:
        pickle.dump(history, file)
    model.save(path + name + '.keras')

def load_model(type, name):
    path = './models/' + type + '/'
    with open(path + name + '_history.pkl', 'rb') as file:
        history = pickle.load(file)
    model = keras_load_model(path + name + '.keras')
    return history, model

def save_training_data(name, X_train, X_test, y_train, y_test, X_train_pad, X_test_pad, y_train_cat, y_test_cat, class_names):
    path = './data/' + name
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(path + '/X_train.pkl', 'wb') as file:
        pickle.dump(X_train, file)
    with open(path + '/X_test.pkl', 'wb') as file:
        pickle.dump(X_test, file)
    with open(path + '/y_train.pkl', 'wb') as file:
        pickle.dump(y_train, file)
    with open(path + '/y_test.pkl', 'wb') as file:
        pickle.dump(y_test, file)
    with open(path + '/X_train_pad.pkl', 'wb') as file:
        pickle.dump(X_train_pad, file)
    with open(path + '/X_test_pad.pkl', 'wb') as file:
        pickle.dump(X_test_pad, file)
    with open(path + '/y_train_cat.pkl', 'wb') as file:
        pickle.dump(y_train_cat, file)
    with open(path + '/y_test_cat.pkl', 'wb') as file:
        pickle.dump(y_test_cat, file)
    with open(path + '/class_names.pkl', 'wb') as file:
        pickle.dump(class_names, file)

def load_training_data(name):
    path = './data/' + name
    with open(path + '/X_train.pkl', 'rb') as file:
        X_train = pickle.load(file)
    with open(path + '/X_test.pkl', 'rb') as file:
        X_test = pickle.load(file)
    with open(path + '/y_train.pkl', 'rb') as file:
        y_train = pickle.load(file)
    with open(path + '/y_test.pkl', 'rb') as file:
        y_test = pickle.load(file)
    with open(path + '/X_train_pad.pkl', 'rb') as file:
        X_train_pad = pickle.load(file)
    with open(path + '/X_test_pad.pkl', 'rb') as file:
        X_test_pad = pickle.load(file)
    with open(path + '/y_train_cat.pkl', 'rb') as file:
        y_train_cat = pickle.load(file)
    with open(path + '/y_test_cat.pkl', 'rb') as file:
        y_test_cat = pickle.load(file)
    with open(path + '/class_names.pkl', 'rb') as file:
        class_names = pickle.load(file)

    return X_train, X_test, y_train, y_test, X_train_pad, X_test_pad, y_train_cat, y_test_cat, class_names