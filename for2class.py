import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix, precision_score, recall_score, f1_score
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings('ignore')

#  (Loading Data)

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
    y_raw = df[target_col]
    

    if y_raw.dtype == 'object':
        mapping = {
            'STANDING': 0, 'SITTING': 0, 'LAYING': 0,
            'WALKING': 1, 'WALKING_DOWNSTAIRS': 1, 'WALKING_UPSTAIRS': 1
        }
    else:
        mapping = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0}
        
    y = y_raw.map(mapping).fillna(0)
    return X, y

X_train_raw, y_train = prepare_data(train_df)
X_test_raw, y_test = prepare_data(test_df)

print(f"Training Samples: {X_train_raw.shape[0]}")
print(f"Original Features: {X_train_raw.shape[1]}")


#  (Normalization)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)


# (PCA)


pca = PCA(n_components=2) 
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("PCA applied. Features reduced to 2 for visualization.")


#  (Logistic Regression)

print("\n--- Logistic Regression Analysis ---")

model_log = LogisticRegression(solver='liblinear', C=1.0)
model_log.fit(X_train_pca, y_train)

y_pred_log = model_log.predict(X_test_pca)
acc_log = accuracy_score(y_test, y_pred_log)
cost_log = log_loss(y_test, model_log.predict_proba(X_test_pca))

print(f"Logistic Accuracy: {acc_log:.4f}")
print(f"Logistic Cost (Log Loss): {cost_log:.4f}")


train_sizes, train_scores, test_scores = learning_curve(
    model_log, X_train_pca, y_train, cv=5, scoring='accuracy', n_jobs=1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Score")
plt.plot(train_sizes, test_mean, 's-', color="green", label="Validation Score")
plt.title("Plot 1: Learning Curve (Check for Overfitting)")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()


#  SVM &  (Decision Boundary)

print("\n--- SVM Analysis ---")

model_svm = SVC(kernel='rbf', C=1.0, probability=True)
model_svm.fit(X_train_pca, y_train)

y_pred_svm = model_svm.predict(X_test_pca)
acc_svm = accuracy_score(y_test, y_pred_svm)
cost_svm = log_loss(y_test, model_svm.predict_proba(X_test_pca))

print(f"SVM Accuracy: {acc_svm:.4f}")
print(f"SVM Cost: {cost_svm:.4f}")


plt.figure(figsize=(10, 6))

# ساخت مش‌بندی برای رسم نواحی تصمیم
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))


Z = model_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('blue', 'red')))


plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', edgecolor='k', s=20)
plt.title("Plot 2: SVM Decision Boundary (Separating Hyperplane)")
plt.xlabel("PC1")
plt.ylabel("PC2")


legend_elements = [Line2D([0], [0], marker='o', color='w', label='Static (0)', markerfacecolor='blue', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Dynamic (1)', markerfacecolor='red', markersize=10)]
plt.legend(handles=legend_elements)
plt.show()


print("\n--- Model Comparison ---")


metrics = {
    'Accuracy': [acc_log, acc_svm],
    'Precision': [precision_score(y_test, y_pred_log), precision_score(y_test, y_pred_svm)],
    'Recall': [recall_score(y_test, y_pred_log), recall_score(y_test, y_pred_svm)],
    'F1-Score': [f1_score(y_test, y_pred_log), f1_score(y_test, y_pred_svm)]
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
plt.title('Plot 3: Model Comparison (Metrics)')
plt.xticks(x, labels)
plt.legend()
plt.ylim(0.9, 1.01) 
plt.grid(axis='y')
plt.show()


print("\n--- Training Cost per Epoch ---")

sgd_clf = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, learning_rate='constant', eta0=0.01, random_state=42)

loss_history = []
epochs = 20 

for epoch in range(epochs):
    sgd_clf.partial_fit(X_train_pca, y_train, classes=np.unique(y_train))
 
    y_prob = sgd_clf.predict_proba(X_train_pca)
    loss = log_loss(y_train, y_prob)
    loss_history.append(loss)


plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, marker='o', color='purple')
plt.title('Plot 4: Cost Function per Epoch (SGD Training)')
plt.xlabel('Epoch')
plt.ylabel('Cost (Log Loss)')
plt.grid()
plt.show()


print("\n--- Unsupervised K-Means ---")
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_test_pca) 


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', s=20)
plt.title('Ground Truth (Actual Labels)')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 2, 2)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=20)
plt.title('Plot 5: K-Means Clustering (Unsupervised)')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.tight_layout()
plt.show()


print("\n--- Generating 3D PCA Plot ---")

pca_3d = PCA(n_components=3)
X_test_pca_3d = pca_3d.fit_transform(X_test_scaled) 

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


scatter = ax.scatter(X_test_pca_3d[:, 0], X_test_pca_3d[:, 1], X_test_pca_3d[:, 2], 
                     c=y_test, cmap='coolwarm', s=20, edgecolor='k', alpha=0.6)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Plot 6: 3D Visualization of Dimension Reduction\n(Separation of Static vs Dynamic)')


legend_elements = [Line2D([0], [0], marker='o', color='w', label='Static (0)', markerfacecolor='blue', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Dynamic (1)', markerfacecolor='red', markersize=10)]
ax.legend(handles=legend_elements)

plt.show()
print("All plots generated successfully.")