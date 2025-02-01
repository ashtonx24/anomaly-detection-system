import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ------------------------
# STEP 1: LOAD DATASET & PREPROCESS
# ------------------------
# Define dataset path
dataset_path = "data/kddcup_sample.csv"  # Update with actual path

# Load dataset
df = pd.read_csv(dataset_path)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Select features (excluding target column if applicable)
X = df.iloc[:, :-1]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------
# STEP 2: BUILD & TRAIN AUTOENCODER
# ------------------------
input_dim = X_scaled.shape[1]

# Define Autoencoder architecture
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation="relu")(input_layer)
encoded = Dense(8, activation="relu")(encoded)  # Bottleneck

decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

# Build & compile model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Train the model
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=64, shuffle=True, validation_split=0.2)

# Save the trained model
model_save_path = "models/autoencoder_model_final.h5"
autoencoder.save(model_save_path)

print(f"✅ Model training completed. Saved to: {model_save_path}")
