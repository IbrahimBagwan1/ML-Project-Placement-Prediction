# Student Score Predictor 📊

This project is a machine learning pipeline that predicts students' **math scores** based on a combination of **categorical** and **numerical** features such as gender, race/ethnicity, parental education level, lunch type, and test preparation course. The project is implemented with modular, production-grade practices using Python.

---

## 🚀 Project Structure

```
ml_project/
├── app.py                 # Main entry point for running the pipeline
├── main.py                # Optional alternate entry
├── Dockerfile             # Docker config for containerization
├── requirements.txt       # Python dependencies
├── setup.py               # Package setup
├── src/
│   └── MlProject/
│       ├── components/    # All ML components (trainer, ingestion, etc.)
│       ├── exception/     # Custom exception handling
│       ├── logger/        # Logging config
│       ├── pipelines/     # Pipeline orchestration
│       ├── utils/         # Utility functions
│       └── new/           # Custom modules / feature experimentation
├── artifacts/             # Intermediate data files (auto-created)
└── .dvc/                  # DVC config for versioning
```

---

## 📚 Problem Statement

Given a student's demographic and academic data, predict their **math score** using regression-based machine learning models. This can be used to analyze factors affecting academic performance and design interventions.

---

## 💡 Features Used

| Feature                      | Type        | Description                            |
|------------------------------|-------------|----------------------------------------|
| `gender`                     | Categorical | Male/Female                            |
| `race/ethnicity`             | Categorical | Group A–E                              |
| `parental level of education`| Categorical | Highest education of parents           |
| `lunch`                      | Categorical | Standard / Free-Reduced                |
| `test preparation course`    | Categorical | Completed / None                       |
| `reading score`              | Numerical   | Score (0–100)                          |
| `writing score`              | Numerical   | Score (0–100)                          |
| `math score` (Target)        | Numerical   | Score (0–100)                          |

---

## 🧪 ML Pipeline Stages

1. **Data Ingestion**
  - Load raw CSV
  - Split into train/test
  - Save artifacts for reproducibility

2. **Data Transformation**
  - Handle missing values
  - Encode categorical variables
  - Scale numerical features
  - Combine into final NumPy arrays

3. **Model Training**
  - Train multiple regression models
  - Compare using R² score
  - Save best-performing model

4. **Model Evaluation**
  - Evaluate on test data
  - Log results and errors

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/IbrahimBagwan1/ml_project.git
cd ml_project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# For Linux/macOS
source venv/bin/activate
# For Windows
venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

---

## 🐳 Docker Support (Optional)

You can build and run this project using Docker:

```bash
docker build -t ml_project .
docker run ml_project
```

---

## 📁 Version Control with DVC

This project uses DVC (Data Version Control) for tracking artifacts like datasets and models.

```bash
dvc init
dvc add path/to/artifact
git add .gitignore data.dvc
git commit -m "Track dataset/model with DVC"
```

---

## 📊 Example Output

Once executed, you will see logs like:

```yaml
Best Model: RandomForestRegressor
Train R² Score: 0.95
Test R² Score: 0.91
Model saved to: artifacts/model.pkl
```

---

## 🧠 Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Model Explainability (SHAP, LIME)
- Frontend integration (Flask Web App or Streamlit)
- CI/CD with GitHub Actions

---

## 📌 Author

**Ibrahim Bagwan**  
GitHub: [@IbrahimBagwan1](https://github.com/IbrahimBagwan1)  
Email: ibrahimbagwan@example.com  
