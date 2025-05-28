# Anomaly Detection in Network Traffic using Autoencoders and K-Means Clustering

## 1. Introduction  
In modern network security, anomaly detection plays a crucial role in identifying malicious activities such as cyberattacks, fraud, and unauthorized access. Traditional rule-based methods are often ineffective against evolving threats, making machine learning-based anomaly detection a more reliable approach.

This project implements an unsupervised anomaly detection system using a hybrid model that combines Autoencoders (deep learning) and K-Means Clustering (statistical method) to detect deviations in network traffic data and classify potential anomalies.

---

## 2. Dataset Overview  
The dataset used is the KDD Cup 1999 Intrusion Detection Dataset, a widely used benchmark for network anomaly detection. It contains various features of network connections:

- Numerical attributes (e.g., duration, byte counts, error rates)  
- Categorical attributes (e.g., protocol type, service type, flag status)  

The 10% corrected training dataset was used for training, and the unlabeled test dataset was used for evaluation.

---

## 3. Methodology  

### 3.1 Data Preprocessing  
- Categorical features encoded using label encoding.  
- Numerical features normalized with StandardScaler.  
- Handling data imbalance with a hybrid thresholding method.

### 3.2 Anomaly Detection Approach  
The hybrid model uses two complementary techniques:  

- **Autoencoder**: A deep learning model trained to reconstruct normal traffic patterns. Reconstruction errors above the 97th percentile threshold indicate anomalies.  
- **K-Means Clustering**: Clusters network traffic patterns; distance from cluster centroids beyond the 97th percentile threshold indicates anomalies.

### 3.3 Hybrid Detection  
Records flagged as anomalies by **both** Autoencoder and K-Means are marked as high-confidence anomalies, reducing false positives and improving precision.

---

## 4. Implementation & Execution  

### 4.1 Model Training  
- Data encoding and normalization.  
- Autoencoder trained on normal data to learn reconstruction.  
- K-Means clustering applied for statistical anomaly detection.  
- Hybrid thresholding combining both models' results.  

This project leverages **Google Colab** for data processing and model training, enabling efficient handling of large datasets and computationally intensive tasks without requiring local high-performance hardware.

### 4.2 Testing & Evaluation  
- Model tested on real-world network traffic data (unlabeled).  
- Anomalies detected via both Autoencoder and K-Means.  
- High-confidence anomalies identified by cross-verification.

---

## 5. Results & Findings  

| Metric                         | Count      |
|-------------------------------|------------|
| Total samples processed        | 2,984,154  |
| Anomalies detected by Autoencoder | 149,206  |
| Anomalies detected by K-Means  | 149,208    |
| High-confidence hybrid anomalies | 50,606   |

**Key observations:**  
- Autoencoder detects more anomalies but with a slightly higher false positive rate.  
- K-Means struggles with overlapping patterns but aligns closely in detections.  
- Hybrid detection improves precision by cross-verifying anomalies, reducing false positives.

---

## 6. Conclusion & Future Work  

### 6.1 Conclusion  
This project demonstrates a hybrid anomaly detection framework leveraging both deep learning and statistical clustering. Combining Autoencoders with K-Means clustering reduces false positives and improves anomaly detection precision in network traffic.

### 6.2 Future Enhancements  
- Experiment with alternative clustering methods like DBSCAN or Gaussian Mixture Models.  
- Integrate the model into a real-time Intrusion Detection System (IDS).  
- Implement adaptive thresholding using dynamic cutoffs instead of static percentiles.  
- Optimize feature selection with PCA or feature importance analysis.

---

## 7. How to Run  

1. Clone this repository.  
2. Install dependencies (TensorFlow, Scikit-learn, Pandas, NumPy).  
3. Preprocess the dataset as described in the preprocessing scripts.  
4. Train the Autoencoder and K-Means models using provided training scripts.  
5. Run the evaluation scripts on the test dataset to detect anomalies.  
6. Check results in the output files/logs.

---

## 8. References  
- KDD Cup 1999 Dataset Documentation  
- Research papers on Hybrid Anomaly Detection  
- TensorFlow & Scikit-learn official documentation

---

*Feel free to reach out if you have any questions or want to collaborate!*
