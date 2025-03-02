import os
import sys
import gc
import signal
import glob
import numpy as np
import wfdb
import matplotlib.pyplot as plt

from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from scipy.interpolate import CubicSpline


# Preferowane odprowadzenia
PREFERRED_LEADS = ["ii", "MLII", "II", "ECG1"]
LUDB_PATH = "ludb/data/"
TARGET_FS = 500
WINDOW_SIZE = 2000
BAD_PATIENTS = [7, 34, 95, 104, 111]

# Mapowanie symboli na klasy (0=none, 1=P, 2=QRS, 3=T)
WAVE_MAP = {'p': 1, 'N': 2, 't': 3}  # 0 = none


def resample_signal(signal, orig_fs=500, target_fs=500):
    if orig_fs == target_fs:
        return signal
    time_orig = np.linspace(0, len(signal) / orig_fs, num=len(signal))
    time_target = np.linspace(0, len(signal) / orig_fs, num=int(len(signal) * target_fs / orig_fs))
    interpolator = CubicSpline(time_orig, signal)
    return interpolator(time_target)


def cleanup_resources(signum, frame):
    print("🛑 Przerywanie... zwalniam pamięć!")
    K.clear_session()
    gc.collect()
    sys.exit(0)

def select_best_lead(record):
    if record.p_signal is None or not hasattr(record, 'sig_name'):
        return None
    for lead in PREFERRED_LEADS:
        if lead in record.sig_name:
            print(f"[DEBUG] Wybrano lead: {lead}")

            return record.p_signal[:, record.sig_name.index(lead)]
    print("[DEBUG] Żaden preferowany lead nie został znaleziony.")
    return None

def find_annotation_file(record_name):
    possible_files = glob.glob(os.path.join(LUDB_PATH, record_name + ".*"))
    for lead in PREFERRED_LEADS:
        for file in possible_files:
            if file.endswith(f".{lead}"):
                print(f"[DEBUG] Znaleziono plik adnotacji: {file}")
                return file
    print(f"[DEBUG] Brak pliku adnotacji dla rekordu {record_name}")
    return None

def load_ecg(record_name):
    annotation_file = find_annotation_file(record_name)
    if annotation_file is None:
        print(f"❌ Brak pliku adnotacji dla {record_name}, pomijam...")
        return None, None
    record_path = os.path.join(LUDB_PATH, record_name)
    record = wfdb.rdrecord(record_path)
    ext = annotation_file.split('.')[-1]
    annotation = wfdb.rdann(annotation_file[:-len(ext)-1], extension=ext)
    print(f"[DEBUG] Wczytano ECG rekordu {record_name} z {len(record.p_signal)} próbkami")
    return record, annotation

def load_all_records():
    data_list = []
    for record_id in range(1, 201):
        if record_id in BAD_PATIENTS:
            print(f"[DEBUG] Pomijam pacjenta {record_id}")
            continue
        record_name = str(record_id)
        rec, ann = load_ecg(record_name)
        if rec is not None and ann is not None:
            best_lead = select_best_lead(rec)
            if best_lead is not None:
                data_list.append((best_lead, ann))
    print(f"[DEBUG] Łącznie wczytano {len(data_list)} rekordów")
    return data_list

def create_label_array(signal, annotation):
    L = len(signal)
    labels = np.zeros(L, dtype=int)
    i = 0
    while i < len(annotation.symbol):
        # Szukamy sekwencji: '(' <symbol> ')'
        if (annotation.symbol[i] == '(' and
                i+2 < len(annotation.symbol) and
                annotation.symbol[i+2] == ')' and
                annotation.symbol[i+1] in WAVE_MAP):
            wave_class = WAVE_MAP[annotation.symbol[i+1]]
            start_idx = annotation.sample[i]
            end_idx   = annotation.sample[i+2]
            start_idx = max(0, start_idx)
            end_idx   = min(L-1, end_idx)
            labels[start_idx:end_idx+1] = wave_class
            i += 3
        else:
            i += 1
    binc = np.bincount(labels, minlength=4)
    print(f"[DEBUG] Rozkład etykiet: none={binc[0]}, P={binc[1]}, QRS={binc[2]}, T={binc[3]}")
    return labels

def generate_training_fragments(signal, labels, num_fragments=5):
    L = len(signal)
    start_min = 1000
    start_max = L - 1000 - WINDOW_SIZE
    if start_max <= start_min:
        return [], []
    X_segments = []
    Y_segments = []
    for _ in range(num_fragments):
        start_idx = randint(start_min, start_max)
        end_idx = start_idx + WINDOW_SIZE
        X_seg = signal[start_idx:end_idx].copy()
        Y_seg = labels[start_idx:end_idx].copy()
        # Normalizacja
        X_seg = (X_seg - np.mean(X_seg)) / (np.std(X_seg) + 1e-8)
        X_segments.append(X_seg)
        Y_segments.append(Y_seg)
    return X_segments, Y_segments

def build_unet(input_length):
    inputs = Input(shape=(input_length,1))
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
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=[
                      'accuracy',
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall')
                  ])
    print("[DEBUG] UNet zbudowany i skompilowany")
    return model

def plot_confusion_matrix_samples(model, X, Y_onehot):
    preds = model.predict(X)
    pred_labels = np.argmax(preds, axis=-1).flatten()
    true_labels = np.argmax(Y_onehot, axis=-1).flatten()

    cm = confusion_matrix(true_labels, pred_labels, labels=[0,1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["none","P","QRS","T"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (sample-level)")
    plt.show()

    print("\n[DEBUG] Classification Report (sample-level):")
    print(classification_report(true_labels, pred_labels,
                                labels=[0,1,2,3],
                                target_names=["none","P","QRS","T"]))


# ================== DODATKOWE FUNKCJE ONSET/OFFSET ==================
def extract_segments_from_prediction(pred_labels):
    segments = []
    current_class = pred_labels[0]
    start = 0
    for i in range(1, len(pred_labels)):
        if pred_labels[i] != current_class:
            if current_class != 0:
                segments.append((start, i - 1, current_class))
            current_class = pred_labels[i]
            start = i
    if current_class != 0:
        segments.append((start, len(pred_labels) - 1, current_class))
    return segments

def evaluate_onset_offset(true_segments, pred_segments, tolerance=150):
    used_pred = set()
    TP = 0
    for (true_start, true_end, true_class) in true_segments:
        found_match = False
        for j, (pred_start, pred_end, pred_class) in enumerate(pred_segments):
            if j in used_pred:
                continue
            if pred_class == true_class:
                onset_diff = abs(pred_start - true_start)
                offset_diff = abs(pred_end - true_end)
                if (onset_diff <= tolerance) and (offset_diff <= tolerance):
                    TP += 1
                    used_pred.add(j)
                    found_match = True
                    break
        # brak dopasowania => FN (liczony niżej)
    FP = len(pred_segments) - len(used_pred)
    FN = len(true_segments) - TP
    return TP, FP, FN

def evaluate_onset_offset_for_dataset(model, X, Y, tolerance=150):
    """
    Przechodzi po wszystkich fragmentach w X, Y.
    Dla każdego:
      - pred_labels = argmax(model.predict(X[i]))
      - true_labels = argmax(Y[i])
      - wyodrębnij segmenty, porównaj onset/offset
    Zwraca 3 liczby: sumaryczne TP, FP, FN.
    """
    preds = model.predict(X)
    total_TP = 0
    total_FP = 0
    total_FN = 0
    for i in range(len(X)):
        pred_labels = np.argmax(preds[i], axis=-1)  # (2000,)
        true_labels = np.argmax(Y[i], axis=-1)      # (2000,)

        pred_segments = extract_segments_from_prediction(pred_labels)
        true_segments = extract_segments_from_prediction(true_labels)

        TP, FP, FN = evaluate_onset_offset(true_segments, pred_segments, tolerance=tolerance)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    return total_TP, total_FP, total_FN


def main():
    signal.signal(signal.SIGINT, cleanup_resources)

    print("[DEBUG] Rozpoczynam wczytywanie rekordów...")
    all_data = load_all_records()
    print(f"[DEBUG] Załadowano {len(all_data)} rekordów.")

    X_fragments = []
    Y_fragments = []
    for idx, (signal_ecg, ann) in enumerate(all_data):
        print(f"[DEBUG] Procesuję pacjenta idx={idx}, sygnał shape={signal_ecg.shape}")
        labels_full = create_label_array(signal_ecg, ann)
        X_segs, Y_segs = generate_training_fragments(signal_ecg, labels_full, num_fragments=10)
        print(f"[DEBUG] Pacjent idx={idx}: wygenerowano {len(X_segs)} fragmentów")
        X_fragments.extend(X_segs)
        Y_fragments.extend(Y_segs)

    print(f"[DEBUG] Łącznie fragmentów: {len(X_fragments)}")
    all_labels = np.concatenate(Y_fragments)
    binc = np.bincount(all_labels, minlength=4)
    print(f"[DEBUG] Globalny rozkład etykiet: none={binc[0]}, P={binc[1]}, QRS={binc[2]}, T={binc[3]}")

    X_fragments = np.array(X_fragments, dtype=np.float32).reshape(-1, WINDOW_SIZE, 1)
    Y_fragments = np.array(Y_fragments, dtype=np.int32)

    Y_onehot = np.zeros((len(Y_fragments), WINDOW_SIZE, 4), dtype=np.float32)
    for i in range(len(Y_fragments)):
        Y_onehot[i, np.arange(WINDOW_SIZE), Y_fragments[i]] = 1.0

    print("[DEBUG] X_fragments shape:", X_fragments.shape)
    print("[DEBUG] Y_onehot shape:", Y_onehot.shape)

    X_train, X_val, y_train, y_val = train_test_split(X_fragments, Y_onehot, test_size=0.2, random_state=42)
    print(f"[DEBUG] Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

    train_labels = np.argmax(y_train, axis=-1).ravel()
    val_labels = np.argmax(y_val, axis=-1).ravel()
    print(f"[DEBUG] Rozkład klas w TRAIN: {np.bincount(train_labels, minlength=4)}")
    print(f"[DEBUG] Rozkład klas w VAL:   {np.bincount(val_labels, minlength=4)}")

    print("[DEBUG] Buduję model UNet...")
    model = build_unet(WINDOW_SIZE)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
        ModelCheckpoint("unet_ecg.h5", monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    print("[DEBUG] Rozpoczynam trening modelu...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # Wizualizacja straty
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    print("[DEBUG] Obliczam macierz konfuzji (sample-level) i classification report...")
    plot_confusion_matrix_samples(model, X_val, y_val)

    # ====================== ONSET/OFFSET EVALUATION ======================
    print("\n[DEBUG] Ewaluacja onset/offset z tolerancją 150 ms (val set):")
    total_TP, total_FP, total_FN = evaluate_onset_offset_for_dataset(model, X_val, y_val, tolerance=150)
    precision = total_TP / (total_TP + total_FP + 1e-9)
    recall    = total_TP / (total_TP + total_FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"Onset/Offset (val) => TP={total_TP}, FP={total_FP}, FN={total_FN}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()
