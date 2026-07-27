# %% [markdown]
# <a href="https://colab.research.google.com/github/wecwic/beleg-dl-gruppe-1/blob/main/Belegaufgabe_Gruppe_1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Belegarbeit Deep Learning für sequentielle Prozessdaten SS2023
# 
# 
# 

# %% [markdown]
# # Beleggruppe

# %% [markdown]
# Gruppen_ID = 1
# 
# Namen: (Surname, First Name, Martikelnummer)

# %%
## List of Classes for Classification
selected_axes = ['Acc_X', 'Acc_Y']
selected_classes = ['Gut', 'Mangelschmierung', 'Pitting_V1']

# %% [markdown]
# # 0. Bibliotheken importieren

# %%
# benötigte Bibliotheken importieren
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

%matplotlib inline
import torch

from torch.utils.data import Dataset, DataLoader, Subset

import torch.nn as nn
import torchmetrics

# %% [markdown]
# # 1. Daten erfassen - Daten importieren

# %%
# Für Nutzung in Google Colab: Mount Google Drive to access the dataset

from google.colab import drive
drive.mount('/content/drive')
data_path = '/content/drive/MyDrive/DL_Daten/Datensatz.csv'

# %%
# Lade Datensatz und Ausgabe Zufallszahl
# Für lokale Nutzung, bitte den Pfad zum Datensatz anpassen

# data_path = 'LINK ZUM DATENSATZ'
data_path = r'C:\dev\beleg-dl-gruppe-1\Datensatz.csv'
df = pd.read_csv(data_path)

df.head(-1)

# %% [markdown]
# # 2. Daten Vorverarbeitung

# %%
####################################################################################################################################################################################################################################################### DIESEN BLOCK NICHT ÄNDERN #########################################################################################
#######################################################################################################################################################################################

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def data_preprocessing(selected_axes=['Acc_X', 'Acc_Y', 'Acc_Z'], selected_classes=['Gut', 'Mangelschmierung', 'Pitting_V1', 'Pitting_V2', 'Pitting_V3', 'Pitting_V4'], x_len=2500):  # Select which Acc-Data should be included axis: ['Acc_X', 'Acc_Y', 'Acc_Z']
    file_path = r'd:\Nutzer\boos\_Arbeit\3. IWM-Prozesskette\7. Datensaetze\Boschbett\\PS_total_10000.csv'
    df = pd.read_csv(file_path)
    df = df.drop(df.columns[0:2], axis=1)
    df = df[df['Status'].isin(selected_classes)]

    maxlen = df['run_ID'].value_counts().max()

    if len(selected_classes) < 6:
        label_arr_old = df['label'].values
        label_arr_new = np.zeros(shape=label_arr_old.shape)
        label_arr = np.vstack((label_arr_old, label_arr_new)).T
        i = int(0)
        for idx in np.unique(label_arr_old):
            label_arr[np.where(label_arr[:, 0] == idx), 1] = i
            i += 1
        label_arr = label_arr.astype(int)
        df['label'] = df['label'].map(dict(label_arr))

    n_classes = len(df['label'].value_counts())
    Load = df[['label', 'run_ID']]
    drop_cols = ['label', 'Status', ]
    drop_cols = list(set(drop_cols) & set(df.columns))  # Check if cols are part of df and select only those
    df = df.drop(columns=drop_cols).reset_index(drop=True)
    input_column_names = selected_axes
    x_columns = input_column_names + ['run_ID']
    g = df.groupby('run_ID').cumcount()
    X = (df[x_columns].set_index(['run_ID', g])
         .unstack(fill_value=0)
         .stack().groupby(level=0)
         .apply(lambda x: x.values)
         .to_numpy())
    X = np.array([i for i in X])
    if x_len is not None:
        X = X[:, 0:x_len, :]
        maxlen = x_len
    # del df  <- in case of memory issues
    g = Load.groupby('run_ID').cumcount()
    y = (Load.set_index(['run_ID', g])
         .unstack(fill_value=0)
         .stack().groupby(level=0)
         .apply(lambda x: x.values[0])
         .to_numpy().astype("int32"))
    y = to_categorical(y, n_classes)
    return X, y, maxlen

# %% [markdown]
# # 3. Daten für Modell vorbereiten - DataSet Class, Aufteilung in X und y, Normierung

# %%
## Define DataSet Class
class DataSet(Dataset):
    # load the dataset
    def __init__(self, X, y):
        self.X = X
        self.y = y

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.y[idx]).float()

# %%
## Data Preprocessing
X, y, maxlen = data_preprocessing(selected_axes=selected_axes,
                                  selected_classes=selected_classes)

## Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    random_state=random.randint(2, 128))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.33,
                                                  shuffle=True,
                                                  random_state=random.randint(2, 128))

## Normalize Data
scaler = StandardScaler().fit(np.vstack(X_train))
X_train = scaler.transform(np.vstack(X_train)).reshape(len(X_train), X[0].shape[0], -1)
X_val = scaler.transform(np.vstack(X_val)).reshape(len(X_val), X[0].shape[0], -1)
X_test = scaler.transform(np.vstack(X_test)).reshape(len(X_test), X[0].shape[0], -1)

## Declare Datasets
train_dataset = DataSet(X_train, y_train)
val_dataset = DataSet(X_val, y_val)
test_dataset = DataSet(X_test, y_test)

# Define DataLoaders
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %% [markdown]
# # Aufgabenstellung

# %%
"""
Entwickeln Sie ein DL-Modell, welches für die Multiklassen-Klassifikation geeignet ist. Zum Bestehen des Belegaufgabe reicht es ein funktionierendes Modell zu entwickeln, welches eine Test-Accuracy von mindestens 55% aufweist. Nutzen Sie alle im Rahmen der Übung vorgestellten Methoden zur Entwicklung und Verbesserung des Modells. Die gewählte Lösungsstrategie sowie die Modellarchitektur und der Programmcode bestimmen die Note, weniger die Höhe der Modellgenauigkeit (solange diese über 55% liegt). Daher bitte ich Sie jeden Lösungsschritt klar zu dokumentieren und falls nötig zu kommentieren. Der bereitgestellte Code dient als Grundlage und kann beliebig erweitert werden. Lediglich die Funktionen "data_preprocessing" und "to_categorical" dürfen nicht verändert werden, sowie die belegspezifischen Variablen selected_axes & selected_classes.

Zur Abgabe reichen Sie bitte dieses ausgefüllte Jupyter-Notebook ein. Sollten Sie weiteren Dokumente (Notebooks, Word-/ Excel-/ TXT-Dateien) befügen wollen, können Sie das gerne vornehmen.

Die Abgabe der Belegarbeit erfolgt zur Deadline im angelegten TU-Cloud Ordner der zugehörigen Gruppe.

Hinweise:
- Es handelt sich um eine Multiklassen-Klassifikation mit 3 Klassen, keine Binärklassifikation!
--> siehe "selected_classes"
--> Achten Sie bitte auf die Anzahl der Neuronen im Output-Layer & Nutzen Sie den One-Hot-Encodern
(siehe https://towardsdatascience.com/target-encoding-for-multi-class-classification-c9a7bcb1a53)
--> Achten Sie bitte auf die Loss-Funktion 
--> Achten Sie bitte auf die Aktivierungsfunktion im Output-Layer
- der Datensatz baut sich aus mehreren Features auf (Beschleunigungsrichtungen --> siehe "selected_axes")
- Nutzen Sie die in der Übung vorgestellten Tuning Methoden zur Verbesserung des Modells
- Nutzen Sie Optuna für die Hyperparameteroptimierung, falls notwendig

- Ich empfehle für die Entwicklung des Modells eine IDE zu verwenden (z.B. PyCharm, Spyder, etc.) und nicht ausschließlich das Jupyter-Notebook.
"""

# %%



