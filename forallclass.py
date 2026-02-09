import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix, precision_score, recall_score, f1_score
from mpl_toolkits.mplot3d import Axes3D 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings('ignore')

try:
    print("Loading datasets...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Error: train.csv or test.csv not found.")
    exit()

def prepare_data(df):
    target_col = df.columns[-1] 
    if 'subject' in df.columns:
        X = df.drop(['subject', target_col], axis=1)
    else:
        X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

X_train_raw, y_train_raw = prepare_data(train_df)
X_test_raw, y_test_raw = prepare_data(test_df)


encoder = LabelEncoder()
all_labels = pd.concat([y_train_raw, y_test_raw])
encoder.fit(all_labels)
y_train = encoder.transform(y_train_raw)
y_test = encoder.transform(y_test_raw)
class_names = encoder.classes_

print(f"Classes: {class_names}")


#  Normalization

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)


# PCA 

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Features after PCA (Model): {X_train_pca.shape[1]}")


#  Logistic Regression

print("\n--- Logistic Regression ---")
model_log = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
model_log.fit(X_train_pca, y_train)

y_pred_log = model_log.predict(X_test_pca)
acc_log = accuracy_score(y_test, y_pred_log)
cost_log = log_loss(y_test, model_log.predict_proba(X_test_pca))

print(f"Accuracy: {acc_log:.4f}")
print(f"Cost (Test Set): {cost_log:.4f}")

# Plot 1: Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    model_log, X_train_pca, y_train, cv=3, scoring='accuracy', n_jobs=1
)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Score")
plt.plot(train_sizes, test_mean, 's-', color="green", label="Validation Score")
plt.title("Plot 1: Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()


#  SVM

print("\n--- SVM Analysis ---")
model_svm = SVC(kernel='rbf', C=1.0, probability=True, decision_function_shape='ovr')
model_svm.fit(X_train_pca, y_train)

y_pred_svm = model_svm.predict(X_test_pca)
acc_svm = accuracy_score(y_test, y_pred_svm)
cost_svm = log_loss(y_test, model_svm.predict_proba(X_test_pca))

print(f"Accuracy: {acc_svm:.4f}")
print(f"Cost (Test Set): {cost_svm:.4f}")

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Plot 2: SVM Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45)
plt.show()


#  Model Comparison

print("\n--- Model Comparison ---")
metrics = {
    'Accuracy': [acc_log, acc_svm],
    'Precision': [precision_score(y_test, y_pred_log, average='weighted'), precision_score(y_test, y_pred_svm, average='weighted')],
    'Recall': [recall_score(y_test, y_pred_log, average='weighted'), recall_score(y_test, y_pred_svm, average='weighted')],
    'F1-Score': [f1_score(y_test, y_pred_log, average='weighted'), f1_score(y_test, y_pred_svm, average='weighted')]
}

labels = list(metrics.keys())
log_vals = [metrics[k][0] for k in labels]
svm_vals = [metrics[k][1] for k in labels]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, log_vals, width, label='Logistic Regression', color='skyblue')
plt.bar(x + width/2, svm_vals, width, label='SVM', color='orange')
plt.ylabel('Score')
plt.title('Plot 3: Model Comparison')
plt.xticks(x, labels)
plt.legend(loc='lower right')
plt.ylim(0.8, 1.0) 
plt.grid(axis='y')
plt.show()


# Cost per Epoch

print("\n--- Training Cost per Epoch ---")

sgd_clf = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, learning_rate='constant', eta0=0.001, random_state=42)

loss_history = []
epochs = 10 

for epoch in range(epochs):
    try:
        sgd_clf.partial_fit(X_train_pca, y_train, classes=np.unique(y_train))
        y_prob = sgd_clf.predict_proba(X_train_pca)
        loss = log_loss(y_train, y_prob)
        loss_history.append(loss)
    except:
        break

if len(loss_history) > 0:
    print(f"Final Training Cost (Last Epoch): {loss_history[-1]:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='purple')
plt.title('Plot 4: Cost Function per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost (Log Loss)')
plt.grid()
plt.show()


#  Unsupervised K-Means

print("\n--- K-Means ---")
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X_test_pca)

print(f"K-Means Cost (Inertia): {kmeans.inertia_:.4f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='nipy_spectral', s=15, alpha=0.7)
plt.title('Ground Truth (2D PCA View)')

plt.subplot(1, 2, 2)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=kmeans.labels_, cmap='nipy_spectral', s=15, alpha=0.7)
plt.title('K-Means Clustering')
plt.show()


#  3D Visualization

print("\n--- Generating 3D Plot ---")
pca_3d = PCA(n_components=3)
X_test_3d = pca_3d.fit_transform(X_test_scaled)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], 
                     c=y_test, cmap='nipy_spectral', s=20, alpha=0.8)

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_title('Plot 6: 3D Visualization of 6 Activities')

legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Classes")
ax.add_artist(legend1)

plt.show()


#  SVM Decision Boundary (2D Visualization)

print("\n--- Generating SVM Decision Boundary Plot ---")

pca_vis = PCA(n_components=2)
X_train_vis = pca_vis.fit_transform(X_train_scaled)

svm_vis = SVC(kernel='rbf', C=1.0)
svm_vis.fit(X_train_vis, y_train)


x_min, x_max = X_train_vis[:, 0].min() - 1, X_train_vis[:, 0].max() + 1
y_min, y_max = X_train_vis[:, 1].min() - 1, X_train_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 8))

plt.contourf(xx, yy, Z, alpha=0.3, cmap='nipy_spectral')


X_test_vis = pca_vis.transform(X_test_scaled)
scatter = plt.scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_test, cmap='nipy_spectral', edgecolors='k', s=25)

plt.title('Plot 7: SVM Decision Boundary (6 Classes Separation)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')


legend1 = plt.legend(*scatter.legend_elements(),
                    loc="upper right", title="Activities")
plt.show()

print("All plots generated successfully.")