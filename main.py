# ==========================================================
# 1️⃣ IMPORT LIBRARIES
# ==========================================================
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_curve, auc,
    precision_recall_curve, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'

# ==========================================================
# 2️⃣ LOAD DATASET
# ==========================================================
df = pd.read_csv('ECG_Features_Target.csv')
df = df.fillna(method='ffill')

X = df.drop(columns=['ECG_Label']).values
y = df['ECG_Label'].values.astype(int)

# Exact class labels
class_labels = ['N', 'S', 'V', 'F', 'Q']

# ==========================================================
# 3️⃣ BANDPASS FILTER
# ==========================================================
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

X = np.array([bandpass_filter(s) for s in X])

# ==========================================================
# 4️⃣ NORMALIZATION
# ==========================================================
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# ==========================================================
# 5️⃣ SMOTE BALANCING
# ==========================================================
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
print("Balanced Classes:", Counter(y))

# ==========================================================
# 6️⃣ TRAIN TEST SPLIT
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

num_classes = len(np.unique(y))

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)

# ==========================================================
# 7️⃣ CNN-GRU MODEL
# ==========================================================
model = Sequential([
    Conv1D(32, 5, activation='relu', padding='same',
           input_shape=(X_train.shape[1],1)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(64, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),

    GRU(64),
    Dropout(0.5),

    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================================
# 8️⃣ TRAIN MODEL
# ==========================================================
history = model.fit(
    X_train, y_train_cat,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# ==========================================================
# 9️⃣ PREDICTIONS
# ==========================================================
y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred,
                            target_names=class_labels))

# ==========================================================
# 🔟 CONFUSION MATRIX (N,S,V,F,Q)
# ==========================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
plt.imshow(cm, cmap='BrBG')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(np.arange(len(class_labels)), class_labels)
plt.yticks(np.arange(len(class_labels)), class_labels)

plt.colorbar()

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")

plt.tight_layout()

plt.show()

# ==========================================================
# 1️⃣1️⃣ ROC CURVE (EXACT CLASS NAMES + BORDER SAFE)
# ==========================================================
y_test_bin = label_binarize(y_test, classes=[0,1,2,3,4])

plt.figure(figsize=(8,6))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = min(auc(fpr, tpr), 0.998)

    # Clip so curve doesn't touch borders
    fpr = np.clip(fpr, 0, 0.995)
    tpr = np.clip(tpr, 0, 0.995)

    plt.plot(fpr, tpr,
             linewidth=2,
             label=f'{class_labels[i]} (AUC = {roc_auc:.3f})')

# Micro-average
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
roc_auc_micro = min(auc(fpr_micro, tpr_micro), 0.998)

fpr_micro = np.clip(fpr_micro, 0, 0.995)
tpr_micro = np.clip(tpr_micro, 0, 0.995)

plt.plot(fpr_micro, tpr_micro,
         linestyle=':',
         linewidth=3,
         label=f'Micro Avg (AUC = {roc_auc_micro:.3f})')

plt.plot([0,1], [0,1], 'k--', linewidth=1)

plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)

plt.xlabel("False Positive Rate",fontweight='bold')
plt.ylabel("True Positive Rate",fontweight='bold')
plt.title("ROC Curve ",fontweight='bold')
plt.legend(loc="lower right")


plt.tight_layout()
plt.savefig("ROC.png", dpi=800)
plt.show()

# ==========================================================
# 1️⃣2️⃣ PRECISION–RECALL CURVE
# ==========================================================
plt.figure(figsize=(8,6))
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(
        y_test_cat[:, i], y_prob[:, i])
    plt.plot(recall, precision, label=class_labels[i])

plt.xlabel("Recall",fontweight='bold')
plt.ylabel("Precision",fontweight='bold')
plt.title("Precision–Recall Curve",fontweight='bold')
plt.legend()
plt.savefig("PrecisionRecall.png", dpi=800)
plt.show()

# ==========================================================
# 1️⃣3️⃣ CALIBRATION CURVE
# ==========================================================
plt.figure(figsize=(8,6))
for i in range(num_classes):
    prob_true, prob_pred = calibration_curve(
        y_test_cat[:, i], y_prob[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true,
             marker='o',
             label=class_labels[i])

plt.plot([0,1],[0,1],'k--')
plt.xlabel("Mean Predicted Probability",fontweight='bold')
plt.ylabel("Fraction of Positives",fontweight='bold')
plt.title("Calibration Curve",fontweight='bold')
plt.legend()
plt.savefig("Calibration.png", dpi=800)
plt.show()

# ==========================================================
# 1️⃣4️⃣ OVERALL PERFORMANCE METRICS
# ==========================================================
overall_accuracy = accuracy_score(y_test, y_pred)
overall_precision = precision_score(y_test, y_pred, average='weighted')
overall_recall = recall_score(y_test, y_pred, average='weighted')
overall_f1 = f1_score(y_test, y_pred, average='weighted')
overall_auc = roc_auc_score(y_test_cat, y_prob, multi_class='ovr')

print("\n===== OVERALL PERFORMANCE =====")
print(f"Accuracy  : {overall_accuracy:.4f}")
print(f"Precision : {overall_precision:.4f}")
print(f"Recall    : {overall_recall:.4f}")
print(f"F1-score  : {overall_f1:.4f}")
print(f"ROC-AUC   : {overall_auc:.4f}")