# Hybrid Disease Prediction System 🩺

An intelligent, interpretable system that predicts respiratory illnesses using a combination of fuzzy logic and machine learning models (Decision Tree, Naive Bayes, Random Forest).

## 📊 Dataset

- **Source**: Kaggle (COVID-19, flu, cold, allergy symptoms)
- **Records**: 12,000+ cleaned instances
- **Features**: 20 binary symptoms (yes/no)
- **Target**: Disease type (COVID-19, FLU, COLD, ALLERGY)

## 🧠 Methodology

- Fuzzy logic computes a risk score (0–10) using symptom combinations.
- Machine learning models predict disease class based on training data.
- A hybrid algorithm combines fuzzy risk and symptom matching for diagnosis.

## 🛠 Technologies

- Python, Pandas, Scikit-learn
- scikit-fuzzy (fuzzy logic)
- Matplotlib (ROC curve, fuzzy risk plot)

## 🧪 Example Usage

User answers 20 yes/no questions → System predicts:
Most likely condition: Flu [Score: 4.5]


## 📄 Files

- `disease_prediction.py` – Main code for prediction and evaluation
- `large_data.csv` – Dataset of symptoms and disease labels
- `documentation.docx` – Detailed project report (methods, results, discussion)

## 📈 Results Summary

| Model           | Accuracy | F1-Score | AUC (ROC) |
|-----------------|----------|----------|-----------|
| Decision Tree   | 0.923    | 0.4804   | 0.99      |
| Naive Bayes     | 0.9342   | 0.815    | 0.98      |
| Random Forest   | 0.8858   | 0.5723   | 0.99      |
| Fuzzy Logic     | 0.4185   | 0.369    | 0.71      |

## 📄 Project Documentation

👉 [Download Full Report](./documentation.docx)

## 🔮 Future Work

- Add symptom severity scoring
- Deploy as a web or mobile app
- Integrate real-time symptom tracking
