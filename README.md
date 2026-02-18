# Restaurant Review Classification (NLP)

## Overview
This project builds a **machine learning system** that automatically classifies restaurant reviews into operational categories such as:

- service  
- food_quality  
- price_value  
- ambience  
- cleanliness  
- management  
- waiting_time  
- order_accuracy  
- staffing  

The goal is to convert unstructured customer feedback into **actionable insights** for restaurant operations.

---

## Features

- Text classification using TF-IDF + LinearSVC  
- Handles imbalanced dataset (resampling + class weights)  
- Model versioning (timestamped models)  
- Experiment tracking (metrics.csv)  
- REST API (FastAPI) for prediction  
- Interactive chat-style UI (Streamlit)  
- Top-3 label predictions with confidence scores  

---

## Model Details

- **Vectorizer:** TF-IDF (unigrams + bigrams)  
- **Model:** LinearSVC  
- **Class Handling:** class_weight="balanced" + controlled resampling  

### Example

**Input**
```
Food was cold and bland.
```

**Output**
```
food_quality
```

## Performance

| Metric      | Value |
|------------|------|
| Accuracy   | ~75% |
| Macro F1   | ~0.68 |

> Performance varies across classes due to imbalance and overlapping categories (e.g., service vs waiting_time).

---

## Project Structure

restaurant-ml/
│
├──images/
│ └── chatapp.png
│
├── data/
│ └── yelp_labeled.csv
│ └── yelp_sample.csv
│
├── models/
│ ├── text_classifier.joblib
│ ├── text_classifier_*.joblib
│ └── metrics.csv
│
├── src/
│ ├── train_model.py
│ ├── predict.py
│ ├── api.py
│ ├── kpi_analysis.py
│ ├── label_data.py
│ ├── load_yelp.py
│ └── chat_app.py
│ 
├── requirements.txt
├── README.md
└── .gitignore

## Installation
 
Clone the repository

```bash
git clone https://github.com/aaronthy/restaurant-review-classification
cd restaurant-review-classification

```

## Create Environment

```bash
python -m venv venv
venv\Scripts\activate

```

## Install Dependencies

```bash
pip install -r requirements.txt

```

## Train Model

```bash
python .\src\train_model.py

```

This will:

-Train the model

-Save versioned models

-Update latest model

-Log metrics

---

## Predict (CLI)

```bash
python src/predict.py

```

## Run API

```bash
uvicorn src.api:app --reload

```

Open API docs:

```bash

http://127.0.0.1:8000/docs

```

## Run Web App

```bash
streamlit run src/chat_app.py

```
Features:

Chat-style input

Real-time predictions

Top 3 label suggestions

---

## Demo

### Streamlit Chat App
![Chat App](images/Screenshot2026-02-18180202.png)
![Chat App](images/Screenshot2026-02-18180522.png)

## Challenges & Learnings

Class imbalance:
Majority "service" class required downsampling and class weighting

Label ambiguity:
Some categories overlap (e.g., service vs waiting_time)

Data quality matters:
Improving labeling consistency has a bigger impact than model changes

---

## Future Improvements

Add more labeled data for rare classes

Improve label definitions

Add probability calibration

Deploy as public web app

Add dashboard for insights

---

## Business Use Case

This model can help restaurant operators:

Identify common customer complaints

Improve service quality

Monitor operational issues

Make data-driven decisions

---

## Author

Aaron Tsen Heng Yee







