import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans

# ------------------------
# STEP 1: LOAD TRAINED MODEL
# ------------------------
model_path = "models/autoencoder_model_final.h5"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}. Verify the path.")

autoencoder = load_model(model_path)

# ------------------------
# STEP 2: LOAD NEW DATASET FOR DETECTION
# ------------------------
new_data_path = "data/kddcup_sample.csv"  # Update with actual test data path

if not os.path.exists(new_data_path):
    raise FileNotFoundError(f"❌ Data file not found at {new_data_path}. Verify the path.")

df = pd.read_csv(new_data_path)

# Encode categorical features
categorical_columns = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.iloc[:, :-1])

# ------------------------
# STEP 3: RUN AUTOENCODER & K-MEANS
# ------------------------
# Compute reconstruction error
reconstructed = autoencoder.predict(X_scaled)
reconstruction_error = np.mean(np.abs(X_scaled - reconstructed), axis=1)

# Set autoencoder anomaly threshold (97th percentile)
threshold_autoencoder = np.percentile(reconstruction_error, 97)
df["Reconstruction Error"] = reconstruction_error
df["Autoencoder_Anomaly"] = reconstruction_error > threshold_autoencoder

# Run K-Means Clustering
k = 5
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=500)
kmeans.fit(X_scaled)
df["Cluster"] = kmeans.predict(X_scaled)

# Compute distances from centroids
distances = np.linalg.norm(X_scaled - kmeans.cluster_centers_[df["Cluster"]], axis=1)
threshold_kmeans = np.percentile(distances, 97)
df["KMeans_Anomaly"] = distances > threshold_kmeans

# Hybrid anomaly detection (both models)
df["Hybrid_Anomaly"] = df["Autoencoder_Anomaly"] & df["KMeans_Anomaly"]

# ------------------------
# STEP 4: SAVE DETECTION RESULTS
# ------------------------
output_file = "results/final_detected_anomalies.csv"
df.to_csv(output_file, index=False)

print(f"✅ Anomaly Detection Completed! Results saved to: {output_file}")
