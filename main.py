import os
import sys
import gc
import signal
import glob
import numpy as np
import wfdb
import matplotlib.pyplot as plt

from random import randint
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, binary_closing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Preferowane odprowadzenia
PREFERRED_LEADS = ["ii", "MLII", "II", "ECG1"]
LUDB_PATH = "ludb/data/"
TARGET_FS = 500
WINDOW_SIZE = 2000
BAD_PATIENTS = [7, 34, 95, 104, 111]

# Mapowanie symboli na klasy (0=none, 1=P, 2=QRS, 3=T)
WAVE_MAP = {'p': 1, 'N': 2, 't': 3}  # 0 = none

def cleanup_resources(signum, frame):
    print("🛑 Przerywanie... zwalniam pamięć!")
    K.clear_session()
    gc.collect()
    sys.exit(0)

def resample_signal(signal, orig_fs=500, target_fs=500):
    if orig_fs == target_fs:
        return signal
    time_orig = np.linspace(0, len(signal) / orig_fs, num=len(signal))
    time_target = np.linspace(0, len(signal) / orig_fs, num=int(len(signal) * target_fs / orig_fs))
    interpolator = CubicSpline(time_orig, signal)
    return interpolator(time_target)

def select_best_lead(record):
    if record.p_signal is None or not hasattr(record, 'sig_name'):
        return None
    for lead in PREFERRED_LEADS:
        if lead in record.sig_name:
            return resample_signal(record.p_signal[:, record.sig_name.index(lead)], TARGET_FS, TARGET_FS)
    return None

def load_ecg(record_name):
    record_path = os.path.join(LUDB_PATH, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, extension='atr')
    return record, annotation

def create_label_array(signal, annotation):
    L = len(signal)
    labels = np.zeros(L, dtype=int)
    i = 0
    while i < len(annotation.symbol):
        if annotation.symbol[i] == '(' and i+2 < len(annotation.symbol) and annotation.symbol[i+2] == ')':
            wave_class = WAVE_MAP.get(annotation.symbol[i+1], 0)
            start_idx = max(0, annotation.sample[i])
            end_idx = min(L-1, annotation.sample[i+2])
            labels[start_idx:end_idx+1] = wave_class
            i += 3
        else:
            i += 1
    return gaussian_filter1d(labels, sigma=3)

def generate_training_fragments(signal, labels, num_fragments=5):
    L = len(signal)
    start_min, start_max = 1000, L - 1000 - WINDOW_SIZE
    if start_max <= start_min:
        return [], []
    X_segments, Y_segments = [], []
    for _ in range(num_fragments):
        start_idx = randint(start_min, start_max)
        X_seg = (signal[start_idx:start_idx+WINDOW_SIZE] - np.mean(signal)) / (np.std(signal) + 1e-8)
        Y_seg = labels[start_idx:start_idx+WINDOW_SIZE]
        X_segments.append(X_seg)
        Y_segments.append(Y_seg)
    return X_segments, Y_segments

def build_unet(input_length):
    inputs = Input(shape=(input_length, 1))
    conv1 = Conv1D(32, 9, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(64, 9, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(128, 9, activation='relu', padding='same')(pool2)
    up1 = UpSampling1D(size=2)(conv3)
    merge1 = concatenate([up1, conv2], axis=-1)
    conv4 = Conv1D(64, 9, activation='relu', padding='same')(merge1)

    up2 = UpSampling1D(size=2)(conv4)
    merge2 = concatenate([up2, conv1], axis=-1)
    conv5 = Conv1D(32, 9, activation='relu', padding='same')(merge2)

    outputs = Conv1D(4, 1, activation='softmax')(conv5)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    signal.signal(signal.SIGINT, cleanup_resources)
    all_data = [load_ecg(str(i)) for i in range(1, 201) if i not in BAD_PATIENTS]
    X_fragments, Y_fragments = [], []
    for rec, ann in all_data:
        lead = select_best_lead(rec)
        if lead is None:
            continue
        labels = create_label_array(lead, ann)
        X_segs, Y_segs = generate_training_fragments(lead, labels, num_fragments=10)
        X_fragments.extend(X_segs)
        Y_fragments.extend(Y_segs)

    X_fragments = np.array(X_fragments).reshape(-1, WINDOW_SIZE, 1)
    Y_fragments = np.eye(4)[np.array(Y_fragments)]
    X_train, X_val, y_train, y_val = train_test_split(X_fragments, Y_fragments, test_size=0.2)

    model = build_unet(WINDOW_SIZE)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

if __name__ == "__main__":
    main()