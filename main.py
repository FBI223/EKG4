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
MODEL_PATH = "unet_ecg.h5"  # Model UNet
TARGET_FS = 500
WINDOW_SIZE = 2000
BAD_PATIENTS = [7, 34, 95, 104, 111]
BAD_PATIENTS_II = [ 95, 104]

# Mapowanie symboli na klasy (0=none, 1=P, 2=QRS, 3=T)
WAVE_MAP = {'p': 1, 'N': 2, 't': 3}  # 0 = none

# Wczytanie modelu
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Brak pliku modelu {MODEL_PATH}")

print("[INFO] Załadowano model UNet")
model = tf.keras.models.load_model(MODEL_PATH)


def augment_signal(signal):
    """Dodaje realistyczne zakłócenia do sygnału EKG."""
    L = len(signal)

    # 1. Szum Gaussowski (małe zakłócenia elektryczne)
    noise = np.random.normal(0, 0.01, L)

    # 2. Dryft bazowy (symulacja oddychania, niskoczęstotliwościowy trend)
    drift = 0.05 * np.sin(np.linspace(0, 2 * np.pi, L))

    # 3. Zakłócenia od mikro skurczów mięśni (szybkie, losowe zmiany)
    muscle_noise = 0.02 * np.random.randn(L) * np.sin(np.linspace(0, 50 * np.pi, L))

    # 4. Zakłócenia elektryczne 50Hz (lekkie zakłócenia sieci elektrycznej)
    electric_noise = 0.01 * np.sin(2 * np.pi * 50 * np.linspace(0, 1, L))

    # Skalujemy każdą perturbację losowym współczynnikiem w zakresie 0.5x - 1.5x
    noise *= np.random.uniform(0.5, 1.5)
    drift *= np.random.uniform(0.5, 1.5)
    muscle_noise *= np.random.uniform(0.5, 1.5)
    electric_noise *= np.random.uniform(0.5, 1.5)

    # Sumujemy wszystkie zakłócenia razem
    augmented_signal = signal + noise + drift + muscle_noise + electric_noise

    return augmented_signal


def augment_signalv1(signal):
    """Dodaje różne realistyczne zakłócenia do sygnału EKG."""
    L = len(signal)

    # 1. Dodanie szumu Gaussowskiego (małe zakłócenia elektryczne)
    noise = np.random.normal(0, 0.01, L)

    # 2. Dryft bazowy (symulacja oddychania, niskoczęstotliwościowy trend)
    drift = 0.05 * np.sin(np.linspace(0, 2*np.pi, L))

    # 3. Zakłócenia od mikro skurczów mięśni (szybkie, losowe zmiany)
    muscle_noise = 0.02 * np.random.randn(L) * np.sin(np.linspace(0, 50*np.pi, L))

    # 4. Zakłócenia elektryczne 50Hz (lekkie zakłócenia sieci elektrycznej)
    electric_noise = 0.01 * np.sin(2 * np.pi * 50 * np.linspace(0, 1, L))

    # Wybieramy losowe zakłócenie
    perturbation = np.random.choice([0, 1, 2, 3])

    if perturbation == 0:
        augmented_signal = signal + noise
    elif perturbation == 1:
        augmented_signal = signal + drift
    elif perturbation == 2:
        augmented_signal = signal + muscle_noise
    else:
        augmented_signal = signal + electric_noise

    return augmented_signal


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

    return best_lead, annotation  # Zwracamy rzeczywisty sygnał, a nie cały obiekt `Record`


def load_ecgv1(record_name):
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
        if record_id in BAD_PATIENTS_II:
            print(f"[DEBUG] Pomijam pacjenta {record_id}")
            continue
        record_name = str(record_id)
        rec, ann = load_ecg(record_name)
        if rec is not None and ann is not None:
            best_lead = rec  # Już jest numpy.ndarray
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
        return [], [], [], []

    X_segments = []
    Y_segments = []
    X_aug_segments = []
    Y_aug_segments = []

    for _ in range(num_fragments):
        start_idx = randint(start_min, start_max)
        end_idx = start_idx + WINDOW_SIZE

        X_seg = signal[start_idx:end_idx].copy()
        Y_seg = labels[start_idx:end_idx].copy()

        # Normalizacja
        X_seg = (X_seg - np.mean(X_seg)) / (np.std(X_seg) + 1e-8)

        # Tworzymy wersję z zakłóceniami
        X_aug = augment_signal(X_seg)

        X_segments.append(X_seg)
        Y_segments.append(Y_seg)

        X_aug_segments.append(X_aug)
        Y_aug_segments.append(Y_seg)  # Adnotacje pozostają bez zmian!

    return X_segments, Y_segments, X_aug_segments, Y_aug_segments


from tensorflow.keras.layers import ZeroPadding1D, Conv1DTranspose, BatchNormalization

def build_unet(input_length):
    inputs = Input(shape=(input_length, 1))

    # =================== Encoder ===================
    c1 = Conv1D(4, 9, padding="same", activation="relu")(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv1D(4, 9, padding="same", activation="relu")(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling1D(pool_size=2, padding="same")(c1)  # 1/2

    c2 = Conv1D(8, 9, padding="same", activation="relu")(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv1D(8, 9, padding="same", activation="relu")(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling1D(pool_size=2, padding="same")(c2)  # 1/4

    c3 = Conv1D(16, 9, padding="same", activation="relu")(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv1D(16, 9, padding="same", activation="relu")(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling1D(pool_size=2, padding="same")(c3)  # 1/8

    c4 = Conv1D(32, 9, padding="same", activation="relu")(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv1D(32, 9, padding="same", activation="relu")(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling1D(pool_size=2, padding="same")(c4)  # 1/16

    c5 = Conv1D(64, 9, padding="same", activation="relu")(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv1D(64, 9, padding="same", activation="relu")(c5)
    c5 = BatchNormalization()(c5)

    # =================== Decoder ===================
    u4 = Conv1DTranspose(32, 8, strides=2, padding="same")(c5)
    if u4.shape[1] != c4.shape[1]:
        u4 = ZeroPadding1D((0, 1))(u4)  # Dopasowanie wymiarów
    u4 = concatenate([u4, c4])

    c6 = Conv1D(32, 9, padding="same", activation="relu")(u4)
    c6 = BatchNormalization()(c6)
    c6 = Conv1D(32, 9, padding="same", activation="relu")(c6)
    c6 = BatchNormalization()(c6)

    u3 = Conv1DTranspose(16, 8, strides=2, padding="same")(c6)
    if u3.shape[1] != c3.shape[1]:
        u3 = ZeroPadding1D((0, 1))(u3)
    u3 = concatenate([u3, c3])

    c7 = Conv1D(16, 9, padding="same", activation="relu")(u3)
    c7 = BatchNormalization()(c7)
    c7 = Conv1D(16, 9, padding="same", activation="relu")(c7)
    c7 = BatchNormalization()(c7)

    u2 = Conv1DTranspose(8, 8, strides=2, padding="same")(c7)
    if u2.shape[1] != c2.shape[1]:
        u2 = ZeroPadding1D((0, 1))(u2)
    u2 = concatenate([u2, c2])

    c8 = Conv1D(8, 9, padding="same", activation="relu")(u2)
    c8 = BatchNormalization()(c8)
    c8 = Conv1D(8, 9, padding="same", activation="relu")(c8)
    c8 = BatchNormalization()(c8)

    u1 = Conv1DTranspose(4, 8, strides=2, padding="same")(c8)
    if u1.shape[1] != c1.shape[1]:
        u1 = ZeroPadding1D((0, 1))(u1)
    u1 = concatenate([u1, c1])

    c9 = Conv1D(4, 9, padding="same", activation="relu")(u1)
    c9 = BatchNormalization()(c9)
    c9 = Conv1D(4, 9, padding="same", activation="relu")(c9)
    c9 = BatchNormalization()(c9)

    # =================== Output ===================
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
    X_aug_fragments = []
    Y_aug_fragments = []

    for idx, (signal_ecg, ann) in enumerate(all_data):
        print(f"[DEBUG] Procesuję pacjenta idx={idx}, sygnał shape={signal_ecg.shape}")
        labels_full = create_label_array(signal_ecg, ann)

        X_segs, Y_segs, X_aug_segs, Y_aug_segs = generate_training_fragments(signal_ecg, labels_full, num_fragments=10)

        print(f"[DEBUG] Pacjent idx={idx}: wygenerowano {len(X_segs)} fragmentów + {len(X_aug_segs)} augmentowanych")

        X_fragments.extend(X_segs)
        Y_fragments.extend(Y_segs)
        X_aug_fragments.extend(X_aug_segs)
        Y_aug_fragments.extend(Y_aug_segs)

    # Łączymy oba zbiory
    X_total = np.array(X_fragments + X_aug_fragments, dtype=np.float32).reshape(-1, WINDOW_SIZE, 1)
    Y_total = np.array(Y_fragments + Y_aug_fragments, dtype=np.int32)

    print(f"[DEBUG] Łącznie fragmentów: {len(X_total)} (oryginalne + augmentowane)")

    # Konwersja do one-hot encoding
    Y_onehot = np.zeros((len(Y_total), WINDOW_SIZE, 4), dtype=np.float32)
    for i in range(len(Y_total)):
        Y_onehot[i, np.arange(WINDOW_SIZE), Y_total[i]] = 1.0

    # Podział na zbiór treningowy i walidacyjny
    X_train, X_val, y_train, y_val = train_test_split(X_total, Y_onehot, test_size=0.2, random_state=42)

    print(f"[DEBUG] Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

    # Trenowanie modelu
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
        epochs=50,
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
    recall = total_TP / (total_TP + total_FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"Onset/Offset (val) => TP={total_TP}, FP={total_FP}, FN={total_FN}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")



def load_test_data(num_samples=5):
    """Losuje `num_samples` pacjentów i zwraca sygnał EKG + adnotacje."""
    available_patients = [str(i) for i in range(1, 201) if i not in BAD_PATIENTS_II]
    selected_patients = np.random.choice(available_patients, num_samples, replace=False)

    X_test, Y_test, records = [], [], []
    for record_name in selected_patients:
        signal, annotation = load_ecg(record_name)
        if signal is None or annotation is None:
            continue

        labels = create_label_array(signal, annotation)

        # Pobranie fragmentu o długości WINDOW_SIZE
        if len(signal) > WINDOW_SIZE:
            start_idx = np.random.randint(0, len(signal) - WINDOW_SIZE)
            X_segment = signal[start_idx:start_idx + WINDOW_SIZE].copy()
            Y_segment = labels[start_idx:start_idx + WINDOW_SIZE].copy()

            # Normalizacja
            X_segment = (X_segment - np.mean(X_segment)) / (np.std(X_segment) + 1e-8)

            X_test.append(X_segment)
            Y_test.append(Y_segment)
            records.append(record_name)

    X_test = np.array(X_test, dtype=np.float32).reshape(-1, WINDOW_SIZE, 1)
    Y_test = np.array(Y_test, dtype=np.int32)

    return X_test, Y_test, records

def plot_predictions(X_test, Y_test, pred_labels, records):
    """Tworzy dwa wykresy: jeden dla rzeczywistych adnotacji, drugi dla predykcji."""
    os.makedirs("models/v4_sota/predictions", exist_ok=True)

    for i in range(len(X_test)):
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Górny wykres - rzeczywiste adnotacje
        axs[0].plot(X_test[i], label="Sygnał EKG", color="black", linewidth=1)
        axs[0].set_title(f"Pacjent {records[i]} - Oryginalne adnotacje (GT)")
        axs[0].set_ylabel("Amplituda")

        true_p = np.where(Y_test[i] == 1)[0]
        true_qrs = np.where(Y_test[i] == 2)[0]
        true_t = np.where(Y_test[i] == 3)[0]

        axs[0].scatter(true_p, X_test[i][true_p], color="blue", label="GT: P", marker="o", s=40)
        axs[0].scatter(true_qrs, X_test[i][true_qrs], color="red", label="GT: QRS", marker="o", s=40)
        axs[0].scatter(true_t, X_test[i][true_t], color="green", label="GT: T", marker="o", s=40)
        axs[0].legend()
        axs[0].grid()

        # Dolny wykres - predykcja modelu
        axs[1].plot(X_test[i], label="Sygnał EKG", color="black", linewidth=1)
        axs[1].set_title(f"Pacjent {records[i]} - Predykcja modelu")
        axs[1].set_xlabel("Próbki")
        axs[1].set_ylabel("Amplituda")

        pred_p = np.where(pred_labels[i] == 1)[0]
        pred_qrs = np.where(pred_labels[i] == 2)[0]
        pred_t = np.where(pred_labels[i] == 3)[0]

        axs[1].scatter(pred_p, X_test[i][pred_p], color="blue", label="Pred: P", marker="x", s=30)
        axs[1].scatter(pred_qrs, X_test[i][pred_qrs], color="red", label="Pred: QRS", marker="x", s=30)
        axs[1].scatter(pred_t, X_test[i][pred_t], color="green", label="Pred: T", marker="x", s=30)
        axs[1].legend()
        axs[1].grid()

        save_path = f"models/v4_sota/predictions/ecg_prediction_{records[i]}.png"
        plt.savefig(save_path)
        plt.close()

    print(f"[INFO] Zapisano wykresy do folderu 'predictions/'")

def run_model_inference():
    """Wykonuje predykcję i rysuje wykresy."""
    X_test, Y_test, records = load_test_data(num_samples=10)
    preds = model.predict(X_test)
    pred_labels = np.argmax(preds, axis=-1)

    plot_predictions(X_test, Y_test, pred_labels, records)

# Uruchomienie inferencji
if __name__ == "__main__":
    #main()
    run_model_inference()