#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Load dataset
df = pd.read_csv(r"D:\ACADEMICS\4th semester\HMI 8870 Knowledge Representation in Biology and medicine\large_data.csv") 
df.dropna(inplace=True)

# Extract symptom list and diseases
symptom_list = df.columns[:-1].tolist()  # All columns except the last
diseases = df.iloc[:, -1].unique().tolist()  # Unique disease names

# Generate symptom weights based on symptom frequency per disease
symptom_weights = {symptom: {} for symptom in symptom_list}
for disease in diseases:
    subset = df[df.iloc[:, -1] == disease]
    for symptom in symptom_list:
        # Scale symptom frequency [0–1] to weight [0–2]
        weight = subset[symptom].mean() * 2
        symptom_weights[symptom][disease] = round(weight, 2)

# --- Get User Input ---
print("Please answer yes/no for the following symptoms:")
user_input = {}
for symptom in symptom_list:
    response = input(f"{symptom.replace('_', ' ').title()}? ").strip().lower()
    user_input[symptom] = 1 if response.startswith("y") else 0

# --- Calculate raw disease scores ---
disease_scores = {disease: 0 for disease in diseases}
for symptom, present in user_input.items():
    for disease in diseases:
        disease_scores[disease] += present * symptom_weights[symptom].get(disease, 0)

# --- Normalize scores against each disease's max possible score ---
max_possible = {
    disease: sum(symptom_weights[s].get(disease, 0) for s in symptom_list)
    for disease in diseases
}

for disease in diseases:
    if max_possible[disease] > 0:
        normalized = disease_scores[disease] / max_possible[disease]
        disease_scores[disease] = round(normalized * 10, 2)
    else:
        disease_scores[disease] = 0

# --- Get top disease prediction ---
sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
top_disease, top_score = sorted_diseases[0]

# --- Define fuzzy membership functions ---
x = np.arange(0, 11, 1)
risk_low = fuzz.trimf(x, [0, 0, 4])
risk_med = fuzz.trimf(x, [3, 5, 7])
risk_high = fuzz.trimf(x, [6, 10, 10])

# Determine which region the score falls into
if top_score <= 4:
    shape = risk_low
    label = "low"
    color = "blue"
elif top_score <= 7:
    shape = risk_med
    label = "medium"
    color = "orange"
else:
    shape = risk_high
    label = "high"
    color = "green"

# --- Print and Plot Result ---
print(f"\nMost likely condition: {top_disease.capitalize()} [Score: {top_score}]")

plt.figure()
plt.plot(x, risk_low, 'b', label='low')
plt.plot(x, risk_med, 'orange', label='medium')
plt.plot(x, risk_high, 'green', label='high')
plt.fill_between(x, 0, shape, color=color, alpha=0.4)
plt.axvline(x=top_score, color='black', linewidth=2)
plt.title(f"Most Likely Condition: {top_disease.upper()} ({top_score})")
plt.xlabel("Risk")
plt.ylabel("Membership")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Load dataset
df = pd.read_csv(r"D:\ACADEMICS\4th semester\HMI 8870 Knowledge Representation in Biology and medicine\large_data.csv") 
df.dropna(inplace=True)

# Extract symptom list and diseases
symptom_list = df.columns[:-1].tolist()  # All columns except the last
diseases = df.iloc[:, -1].unique().tolist()  # Unique disease names

# Generate symptom weights based on symptom frequency per disease
symptom_weights = {symptom: {} for symptom in symptom_list}
for disease in diseases:
    subset = df[df.iloc[:, -1] == disease]
    for symptom in symptom_list:
        # Scale symptom frequency [0–1] to weight [0–2]
        weight = subset[symptom].mean() * 2
        symptom_weights[symptom][disease] = round(weight, 2)

# --- Get User Input ---
print("Please answer yes/no for the following symptoms:")
user_input = {}
for symptom in symptom_list:
    response = input(f"{symptom.replace('_', ' ').title()}? ").strip().lower()
    user_input[symptom] = 1 if response.startswith("y") else 0

# --- Calculate raw disease scores ---
disease_scores = {disease: 0 for disease in diseases}
for symptom, present in user_input.items():
    for disease in diseases:
        disease_scores[disease] += present * symptom_weights[symptom].get(disease, 0)

# --- Normalize scores against each disease's max possible score ---
max_possible = {
    disease: sum(symptom_weights[s].get(disease, 0) for s in symptom_list)
    for disease in diseases
}

for disease in diseases:
    if max_possible[disease] > 0:
        normalized = disease_scores[disease] / max_possible[disease]
        disease_scores[disease] = round(normalized * 10, 2)
    else:
        disease_scores[disease] = 0

# --- Get top disease prediction ---
sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
top_disease, top_score = sorted_diseases[0]

# --- Define fuzzy membership functions ---
x = np.arange(0, 11, 1)
risk_low = fuzz.trimf(x, [0, 0, 4])
risk_med = fuzz.trimf(x, [3, 5, 7])
risk_high = fuzz.trimf(x, [6, 10, 10])

# Determine which region the score falls into
if top_score <= 4:
    shape = risk_low
    label = "low"
    color = "blue"
elif top_score <= 7:
    shape = risk_med
    label = "medium"
    color = "orange"
else:
    shape = risk_high
    label = "high"
    color = "green"

# --- Print and Plot Result ---
print(f"\nMost likely condition: {top_disease.capitalize()} [Score: {top_score}]")

plt.figure()
plt.plot(x, risk_low, 'b', label='low')
plt.plot(x, risk_med, 'orange', label='medium')
plt.plot(x, risk_high, 'green', label='high')
plt.fill_between(x, 0, shape, color=color, alpha=0.4)
plt.axvline(x=top_score, color='black', linewidth=2)
plt.title(f"Most Likely Condition: {top_disease.upper()} ({top_score})")
plt.xlabel("Risk")
plt.ylabel("Membership")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


#### Decision Tree ###


# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r"D:\ACADEMICS\4th semester\HMI 8870 Knowledge Representation in Biology and medicine\large_data.csv")
df.dropna(inplace=True)

X = df.iloc[:, :-1]  # all symptoms
y = df.iloc[:, -1]   # disease label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

# Results
print("\n--- Decision Tree Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


### Naive Bayes ####


# In[16]:


from sklearn.naive_bayes import GaussianNB

# Train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Results
print("\n--- Naive Bayes Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))


# In[ ]:


### Random Forest #####


# In[17]:


from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Results
print("\n--- Random Forest Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# In[19]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r"D:\ACADEMICS\4th semester\HMI 8870 Knowledge Representation in Biology and medicine\large_data.csv")
df.dropna(inplace=True)

# Features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
diseases = y.unique().tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Fuzzy Logic Scoring System
# ---------------------------

# Build symptom weights (mean frequency * 2)
symptom_weights = {symptom: {} for symptom in X.columns}
for disease in diseases:
    subset = X_train[y_train == disease]
    for symptom in X.columns:
        weight = subset[symptom].mean() * 2
        symptom_weights[symptom][disease] = round(weight, 2)

# Predict function for fuzzy logic
def fuzzy_predict(sample):
    scores = {disease: 0 for disease in diseases}
    max_score = {disease: sum(symptom_weights[s][disease] for s in X.columns) for disease in diseases}
    for disease in diseases:
        for s in X.columns:
            scores[disease] += sample[s] * symptom_weights[s].get(disease, 0)
        if max_score[disease] > 0:
            scores[disease] = scores[disease] / max_score[disease] * 10
    return max(scores, key=scores.get)

# Apply fuzzy logic to test data
y_pred_fuzzy = X_test.apply(fuzzy_predict, axis=1)

# ----------------------------
# ML Model Evaluations
# ----------------------------

def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = classification_report(y_test, y_pred, output_dict=True)["macro avg"]["f1-score"]
    return {"accuracy": round(acc, 4), "f1": round(f1, 4)}

results = {}

# Decision Tree
results["Decision Tree"] = evaluate_model(DecisionTreeClassifier(max_depth=5, random_state=42), "Decision Tree")

# Naive Bayes
results["Naive Bayes"] = evaluate_model(GaussianNB(), "Naive Bayes")

# Random Forest
results["Random Forest"] = evaluate_model(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")

# Fuzzy Logic
fuzzy_accuracy = accuracy_score(y_test, y_pred_fuzzy)
fuzzy_f1 = classification_report(y_test, y_pred_fuzzy, output_dict=True)["macro avg"]["f1-score"]
results["Fuzzy Logic"] = {"accuracy": round(fuzzy_accuracy, 4), "f1": round(fuzzy_f1, 4)}

# ---------------------------
# Print Comparison Results
# ---------------------------

print("\n=== Model Performance Comparison ===")
for model, scores in results.items():
    print(f"\n{model}")
    print(f"Accuracy : {scores['accuracy']}")
    print(f"F1-Score : {scores['f1']}")


# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

# Load dataset
df = pd.read_csv(r"D:\ACADEMICS\4th semester\HMI 8870 Knowledge Representation in Biology and medicine\large_data.csv")
df.dropna(inplace=True)

# Features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
classes = sorted(y.unique().tolist())
y_bin = label_binarize(y, classes=classes)
n_classes = y_bin.shape[1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test_bin = label_binarize(y_test, classes=classes)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- ML Models ----------
models = {
    "Naive Bayes": OneVsRestClassifier(GaussianNB()),
    "Decision Tree": OneVsRestClassifier(DecisionTreeClassifier(max_depth=5, random_state=42)),
    "Random Forest": OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
}

# Plot setup
plt.figure(figsize=(10, 7))

# ML ROC curves
for name, model in models.items():
    model.fit(X_train_scaled, label_binarize(y_train, classes=classes))
    y_score = model.predict_proba(X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), np.array(y_score).ravel())
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_score:.2f})")

# ---------- Fuzzy Logic ----------
# Build symptom weights
symptom_weights = {symptom: {} for symptom in X.columns}
for disease in classes:
    subset = X_train[y_train == disease]
    for symptom in X.columns:
        weight = subset[symptom].mean() * 2
        symptom_weights[symptom][disease] = round(weight, 2)

# Predict fuzzy scores (0–10 scaled per class)
fuzzy_scores = []
for _, row in X_test.iterrows():
    scores = {}
    max_scores = {d: sum(symptom_weights[s][d] for s in X.columns) for d in classes}
    for disease in classes:
        score = sum(row[s] * symptom_weights[s].get(disease, 0) for s in X.columns)
        score = score / max_scores[disease] * 10 if max_scores[disease] > 0 else 0
        scores[disease] = score
    fuzzy_scores.append([scores[d] for d in classes])

# Fuzzy ROC
fuzzy_scores = np.array(fuzzy_scores)
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), fuzzy_scores.ravel())
auc_score = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2, linestyle='--', color='black', label=f"Fuzzy Logic (AUC = {auc_score:.2f})")

# ---------- Final Plot ----------
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison: Fuzzy Logic vs ML Models")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




