**Anomaly Detection in Network Traffic using Autoencoders and K-Means Clustering**

# **1. Introduction**
In modern network security, **anomaly detection** plays a crucial role in identifying malicious activities such as cyberattacks, fraud, and unauthorized access. Traditional rule-based methods are often ineffective against evolving threats, making **machine learning-based anomaly detection** a more reliable approach.

This project explores **unsupervised anomaly detection** using a **Hybrid Model combining Autoencoders and K-Means Clustering**. By leveraging these models, we can detect deviations in network traffic data and classify them as potential anomalies.

# **2. Dataset Overview**
The dataset used in this project is the **KDD Cup 1999 Intrusion Detection Dataset**, one of the most widely used benchmarks for network anomaly detection. It contains network connection records with various features, including:
- **Numerical Attributes** (e.g., duration, byte counts, error rates)
- **Categorical Attributes** (e.g., protocol type, service type, flag status)

We used both the **10% corrected training dataset** for model training and the **unlabeled test dataset** for evaluating real-world performance.

# **3. Methodology**
### **3.1 Data Preprocessing**
- **Categorical Encoding**: Label encoding was applied to categorical features.
- **Feature Scaling**: StandardScaler was used to normalize numerical values.
- **Handling Imbalance**: SMOTE was initially considered but later replaced with a hybrid thresholding method.

### **3.2 Anomaly Detection Approach**
This project implements a **Hybrid Model**, combining two different anomaly detection techniques:

**(i) Autoencoder (Deep Learning-based)**
- Autoencoders learn to reconstruct normal network traffic efficiently.
- Reconstruction error is used to determine anomalies.
- Anomalies are identified using a threshold (97th percentile of reconstruction errors).

**(ii) K-Means Clustering (Statistical-based)**
- Clusters similar network traffic patterns.
- Distance from the cluster centroid is used as an anomaly indicator.
- The 97th percentile of distances is used as the anomaly threshold.

### **3.3 Hybrid Anomaly Detection**
- If **both** Autoencoder and K-Means classify a record as an anomaly, it is marked as a **high-confidence anomaly**.
- This method reduces false positives and increases precision.

# **4. Implementation & Execution**
The implementation follows these structured steps:
### **4.1 Model Training**
1. **Preprocessing**: Data encoding and normalization.
2. **Autoencoder Training**: Deep learning model trained on normal network data.
3. **K-Means Clustering**: Used for anomaly detection via clustering.
4. **Hybrid Thresholding**: Anomalies are detected using both methods.

### **4.2 Testing & Evaluation**
1. **Test Data Processing**: The model is tested on real-world traffic data (unlabeled KDD dataset).
2. **Anomaly Detection Execution**: The trained Autoencoder and K-Means model analyze test samples.
3. **Hybrid Anomaly Flagging**: Combining results from both models.
4. **Result Analysis**: Comparison of detected anomalies and clustering behavior.

# **5. Results & Findings**
✅ **Total Samples Processed**: 2,984,154
✅ **Autoencoder Detected Anomalies**: 149,206
✅ **K-Means Detected Anomalies**: 149,208
✅ **High-Confidence Hybrid Anomalies**: 50,606

**Key Observations:**
- **Autoencoder alone** detects more anomalies but has a slightly higher false positive rate.
- **K-Means alone** identifies similar anomalies but struggles with overlapping patterns.
- **Hybrid detection** improves precision by cross-verifying anomalies, reducing false positives.

# **6. Conclusion & Future Work**
### **6.1 Conclusion**
This project successfully demonstrated a **hybrid anomaly detection framework** for network security. By combining deep learning-based Autoencoders with K-Means clustering, the model improves precision and reduces false positives.

### **6.2 Future Enhancements**
🚀 **Fine-tuning K-Means Clustering**: Experimenting with alternative clustering methods like DBSCAN or Gaussian Mixture Models.
🚀 **Real-Time Deployment**: Integrating the model into a real-time Intrusion Detection System (IDS).
🚀 **Adaptive Thresholding**: Using dynamic thresholds instead of static percentile-based cutoffs.
🚀 **Feature Selection Optimization**: Implementing PCA or feature importance analysis to refine input features.

# **7. References**
- KDD Cup 1999 Dataset Documentation
- Research papers on Hybrid Anomaly Detection
- TensorFlow & Scikit-learn Documentation
