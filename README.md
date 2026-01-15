# Insider Threat Detection using Machine Learning

##  Project Overview
This project focuses on detecting insider threats by analyzing employee email communication patterns using machine learning techniques. The study utilizes the CERT Insider Threat Detection Dataset to identify anomalous or suspicious behaviors that may indicate insider attacks.

The project is implemented as a Jupyter Notebook and follows a complete machine learning pipeline including preprocessing, clustering-based labeling, and supervised model evaluation.

---

##  Dataset
- **Dataset Name:** CERT Insider Threat Detection Dataset
- **File Used:** email.csv
- **Source:** Kaggle

 Dataset Link:  
https://www.kaggle.com/datasets/nitishabharathi/cert-insider-threat

>  The dataset is **not included** in this repository due to licensing and size restrictions.  
> Please download email.csv from Kaggle and place it in the project root directory before running the notebook.

---

##  Project Structure

```
insider-threat-detection/
 CSE427_Project_G07.ipynb          # Main Jupyter Notebook with complete ML pipeline
 README.md                          # Project documentation
─ email.csv                          # Dataset (download from Kaggle)
```

---

##  Project Components

### **1. Data Loading & Exploration**
- Load email dataset from Google Drive/Kaggle
- Exploratory Data Analysis (EDA)
- Missing values analysis
- Visualizations: correlation matrix, emails per user, emails per day, internal vs external emails

### **2. Data Preprocessing**
- Handle missing values (fill cc and cc with 'None')
- Date/time parsing and feature extraction
- Data type validation and cleaning

### **3. Feature Engineering**

**Temporal Features:**
- Hour of email sent
- Day of week
- Is weekend flag
- After-hours email detection

**Email Metrics:**
- Number of recipients (to, cc, bcc)
- Total recipients count
- Is internal email (contains @dtaa.com)
- Content length
- Has attachments flag

**User-Level Aggregations:**
- Average number of recipients
- Internal email ratio
- Average content length
- Attachment ratio
- Total attachments sent
- After-hours email count
- Weekend activity count
- Average active hour & hour std deviation
- After-hours ratio (derived)
- Attachment intensity (derived)
- Unique PC count per user

### **4. Unsupervised Learning (Clustering)**
- **PCA:** Dimensionality reduction to 10 components
- **K-Means Clustering:** 
  - Optimal K selection using Elbow method & Silhouette analysis
  - Identified optimal K = 2
  - Suspicious vs Normal user classification
  - Anomaly scoring based on distance from cluster centers
- **MiniBatchKMeans:** Alternative clustering approach for large datasets

### **5. Supervised Learning (Classification)**

**Models Evaluated:**
1. Decision Tree Classifier
2. Random Forest Classifier (100 estimators)
3. Gradient Boosting Classifier
4. Logistic Regression
5. Support Vector Machine (SVM)
6. K-Nearest Neighbors (KNN)
7. Gaussian Naive Bayes
8. AdaBoost Classifier (100 estimators)

**Evaluation Metrics:**
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report
- Cross-Validation (5-fold)

---

##  Key Findings

- Dataset: ~2.6M emails from multiple users
- Features engineered: 13 behavioral features per user
- Target variable: Binary (Normal/Suspicious)
- Best performing models: Gradient Boosting, Random Forest, AdaBoost
- All models achieved >90% accuracy

---

##  Technologies & Libraries

**Data Processing:**
- pandas, numpy

**ML & Preprocessing:**
- scikit-learn (preprocessing, model_selection, ensemble, linear_model, svm, neighbors, naive_bayes, tree, metrics)

**Visualization:**
- matplotlib, seaborn

**Dimensionality Reduction:**
- PCA (Principal Component Analysis)

**Clustering:**
- K-Means, MiniBatchKMeans

---

##  How to Run

1. **Download the dataset** from Kaggle (email.csv)
2. **Upload to Google Drive** and update the file path in the notebook
3. **Open the Jupyter Notebook:** CSE427_Project_G07.ipynb
4. **Run all cells** in sequence to:
   - Load and explore the data
   - Preprocess and engineer features
   - Perform clustering analysis
   - Train and evaluate classification models

---

##  Output

- **insider_threat_results.csv:** User-level predictions with cluster assignments and threat labels
- **Visualizations:** Elbow plots, silhouette analysis, scatter plots, model performance charts

---

