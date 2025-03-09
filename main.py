import os
import sys
import gc
import signal
import glob
import numpy as np
import wfdb
import matplotlib.pyplot as plt

from random import randint
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# Preferowane odprowadzenia
PREFERRED_LEADS = ["MLII", "II", "ECG1", "mlii", "ii", "ecg1"]
# Ścieżki do baz danych
QTDB_PATH = "C:/Users/msztu/Documents/EKG4/qtdb/"
LUDB_PATH = "ludb/data/"
MODEL_PATH = "unet_ecg.h5"  # Model UNet
TARGET_FS = 500
WINDOW_SIZE = 2000
BAD_PATIENTS = [7, 34, 90, 95, 104, 111]
BAD_PATIENTS_II = []
# Mapowanie symboli na klasy (0=none, 1=P, 2=QRS, 3=T)
WAVE_MAP = {'p': 1, 'N': 2, 't': 3}

# -------------------- Utility & Data Loading Functions --------------------
def cleanup_resources(signum, frame):
    print("🛑 Przerywanie... zwalniam pamięć!")
    K.clear_session()
    gc.collect()
    sys.exit(0)

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
    """Wczytuje sygnał EKG i adnotacje dla danego rekordu."""
    annotation_file = find_annotation_file(record_name)
    if annotation_file is None:
        return None, None
    record_path = os.path.join(LUDB_PATH, record_name)
    record = wfdb.rdrecord(record_path)
    ext = annotation_file.split('.')[-1]
    annotation = wfdb.rdann(annotation_file[:-len(ext)-1], extension=ext)
    best_lead = None
    for lead in PREFERRED_LEADS:
        if lead in record.sig_name:
            best_lead = record.p_signal[:, record.sig_name.index(lead)]
            break
    if best_lead is None:
        return None, None
    return best_lead, annotation

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
            end_idx = annotation.sample[i+2]
            start_idx = max(0, start_idx)
            end_idx = min(L-1, end_idx)
            labels[start_idx:end_idx+1] = wave_class
            i += 3
        else:
            i += 1
    print(f"[DEBUG] Rozkład etykiet: {np.bincount(labels, minlength=4)}")
    return labels

# -------------------- Data Augmentation Functions --------------------
def augment_signal(signal):
    """Dodaje realistyczne zakłócenia do sygnału EKG."""
    L = len(signal)
    noise = np.random.normal(0, 0.01, L) * np.random.uniform(0.5, 1.5)
    drift = 0.05 * np.sin(np.linspace(0, 2 * np.pi, L)) * np.random.uniform(0.5, 1.5)
    muscle_noise = 0.02 * np.random.randn(L) * np.sin(np.linspace(0, 50 * np.pi, L)) * np.random.uniform(0.5, 1.5)
    electric_noise = 0.01 * np.sin(2 * np.pi * 50 * np.linspace(0, 1, L)) * np.random.uniform(0.5, 1.5)
    return signal + noise + drift + muscle_noise + electric_noise

def generate_training_fragments(signal, labels, num_fragments=5):
    L = len(signal)
    start_min = 1000
    start_max = L - 1000 - WINDOW_SIZE
    if start_max <= start_min:
        return [], [], [], []
    X_segments, Y_segments, X_aug_segments, Y_aug_segments = [], [], [], []
    for _ in range(num_fragments):
        start_idx = randint(start_min, start_max)
        end_idx = start_idx + WINDOW_SIZE
        X_seg = signal[start_idx:end_idx].copy()
        Y_seg = labels[start_idx:end_idx].copy()
        # Normalizacja
        X_seg = (X_seg - np.mean(X_seg)) / (np.std(X_seg) + 1e-8)
        # Augmentacja
        X_aug = augment_signal(X_seg)
        X_segments.append(X_seg)
        Y_segments.append(Y_seg)
        X_aug_segments.append(X_aug)
        Y_aug_segments.append(Y_seg)
    return X_segments, Y_segments, X_aug_segments, Y_aug_segments

def load_all_records_ludb():
    data_list = []
    for record_id in range(1, 201):
        if record_id in BAD_PATIENTS_II:
            print(f"[DEBUG] Pomijam pacjenta {record_id}")
            continue
        record_name = str(record_id)
        rec, ann = load_ecg(record_name)
        if rec is not None and ann is not None:
            data_list.append((rec, ann))
    print(f"[DEBUG] Łącznie wczytano {len(data_list)} rekordów")
    return data_list

def generate_fragments_for_cv(all_data, num_fragments=10):
    X_fragments = []
    Y_fragments = []
    groups = []  # identyfikatory pacjentów (indeks rekordu)
    for idx, (signal_ecg, ann) in enumerate(all_data):
        labels_full = create_label_array(signal_ecg, ann)
        X_segs, Y_segs, X_aug_segs, Y_aug_segs = generate_training_fragments(signal_ecg, labels_full, num_fragments)
        X_fragments.extend(X_segs + X_aug_segs)
        Y_fragments.extend(Y_segs + Y_aug_segs)
        groups.extend([idx] * (len(X_segs) + len(X_aug_segs)))
    X_total = np.array(X_fragments, dtype=np.float32).reshape(-1, WINDOW_SIZE, 1)
    Y_total = np.array(Y_fragments, dtype=np.int32)
    return X_total, Y_total, groups

# -------------------- Model Definition --------------------
def build_unet(input_length):
    inputs = Input(shape=(input_length, 1))
    # Encoder
    c1 = Conv1D(4, 9, padding="same", activation="relu")(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv1D(4, 9, padding="same", activation="relu")(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling1D(pool_size=2, padding="same")(c1)

    c2 = Conv1D(8, 9, padding="same", activation="relu")(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv1D(8, 9, padding="same", activation="relu")(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling1D(pool_size=2, padding="same")(c2)

    c3 = Conv1D(16, 9, padding="same", activation="relu")(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv1D(16, 9, padding="same", activation="relu")(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling1D(pool_size=2, padding="same")(c3)

    c4 = Conv1D(32, 9, padding="same", activation="relu")(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv1D(32, 9, padding="same", activation="relu")(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling1D(pool_size=2, padding="same")(c4)

    c5 = Conv1D(64, 9, padding="same", activation="relu")(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv1D(64, 9, padding="same", activation="relu")(c5)
    c5 = BatchNormalization()(c5)

    # Decoder
    u4 = tf.keras.layers.Conv1DTranspose(32, 8, strides=2, padding="same")(c5)
    if u4.shape[1] != c4.shape[1]:
        u4 = tf.keras.layers.ZeroPadding1D((0, 1))(u4)
    u4 = concatenate([u4, c4])

    c6 = Conv1D(32, 9, padding="same", activation="relu")(u4)
    c6 = BatchNormalization()(c6)
    c6 = Conv1D(32, 9, padding="same", activation="relu")(c6)
    c6 = BatchNormalization()(c6)

    u3 = tf.keras.layers.Conv1DTranspose(16, 8, strides=2, padding="same")(c6)
    if u3.shape[1] != c3.shape[1]:
        u3 = tf.keras.layers.ZeroPadding1D((0, 1))(u3)
    u3 = concatenate([u3, c3])

    c7 = Conv1D(16, 9, padding="same", activation="relu")(u3)
    c7 = BatchNormalization()(c7)
    c7 = Conv1D(16, 9, padding="same", activation="relu")(c7)
    c7 = BatchNormalization()(c7)

    u2 = tf.keras.layers.Conv1DTranspose(8, 8, strides=2, padding="same")(c7)
    if u2.shape[1] != c2.shape[1]:
        u2 = tf.keras.layers.ZeroPadding1D((0, 1))(u2)
    u2 = concatenate([u2, c2])

    c8 = Conv1D(8, 9, padding="same", activation="relu")(u2)
    c8 = BatchNormalization()(c8)
    c8 = Conv1D(8, 9, padding="same", activation="relu")(c8)
    c8 = BatchNormalization()(c8)

    u1 = tf.keras.layers.Conv1DTranspose(4, 8, strides=2, padding="same")(c8)
    if u1.shape[1] != c1.shape[1]:
        u1 = tf.keras.layers.ZeroPadding1D((0, 1))(u1)
    u1 = concatenate([u1, c1])

    c9 = Conv1D(4, 9, padding="same", activation="relu")(u1)
    c9 = BatchNormalization()(c9)
    c9 = Conv1D(4, 9, padding="same", activation="relu")(c9)
    c9 = BatchNormalization()(c9)

    outputs = Conv1D(4, 1, activation="softmax")(c9)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    print("[DEBUG] Model UNet zbudowany")
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

# -------------------- Main z Cross Walidacją --------------------
def main():
    signal.signal(signal.SIGINT, cleanup_resources)

    print("[DEBUG] Rozpoczynam wczytywanie rekordów...")
    all_data = load_all_records_ludb()
    print(f"[DEBUG] Załadowano {len(all_data)} rekordów.")

    X_fragments = []
    Y_fragments = []
    groups = []  # identyfikatory pacjentów (indeks rekordu)

    for idx, (signal_ecg, ann) in enumerate(all_data):
        print(f"[DEBUG] Procesuję pacjenta idx={idx}, sygnał shape={signal_ecg.shape}")
        labels_full = create_label_array(signal_ecg, ann)
        X_segs, Y_segs, X_aug_segs, Y_aug_segs = generate_training_fragments(signal_ecg, labels_full, num_fragments=10)
        print(f"[DEBUG] Pacjent idx={idx}: wygenerowano {len(X_segs)} fragmentów + {len(X_aug_segs)} augmentowanych")

        X_fragments.extend(X_segs + X_aug_segs)
        Y_fragments.extend(Y_segs + Y_aug_segs)
        groups.extend([idx] * (len(X_segs) + len(X_aug_segs)))

    X_total = np.array(X_fragments, dtype=np.float32).reshape(-1, WINDOW_SIZE, 1)
    Y_total = np.array(Y_fragments, dtype=np.int32)
    print(f"[DEBUG] Łącznie fragmentów: {len(X_total)} (oryginalne + augmentowane)")

    # One-hot encoding etykiet
    Y_onehot = np.zeros((len(Y_total), WINDOW_SIZE, 4), dtype=np.float32)
    for i in range(len(Y_total)):
        Y_onehot[i, np.arange(WINDOW_SIZE), Y_total[i]] = 1.0

    # Użycie walidacji krzyżowej z grupowaniem (wszystkie fragmenty jednego pacjenta razem)
    gkf = GroupKFold(n_splits=5)
    fold = 0
    for train_idx, val_idx in gkf.split(X_total, Y_onehot, groups):
        print(f"[DEBUG] Fold {fold}: Train size: {len(train_idx)}, Val size: {len(val_idx)}")
        X_train, X_val = X_total[train_idx], X_total[val_idx]
        y_train, y_val = Y_onehot[train_idx], Y_onehot[val_idx]
        model = build_unet(WINDOW_SIZE)
        model.summary()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
            ModelCheckpoint(f"unet_ecg_fold{fold}.h5", monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

        # Wizualizacja strat
        plt.figure()
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.title(f"Loss - Fold {fold}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        print(f"[DEBUG] Ewaluacja modelu dla fold {fold}")
        plot_confusion_matrix_samples(model, X_val, y_val)

        fold += 1

if __name__ == "__main__":
    main()
