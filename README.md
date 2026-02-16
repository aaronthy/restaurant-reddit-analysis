# Restaurant Failure Analysis (NLP Project)

## Overview
This project analyzes large-scale restaurant review data to identify key factors that contribute to poor customer satisfaction and potential business failure.

Using natural language processing (NLP) and data analysis techniques, the system extracts insights from customer reviews and ranks operational issues based on their impact.

---

## Problem Statement
Restaurant businesses often struggle to understand the root causes of customer dissatisfaction. This project aims to answer:

"What factors most strongly drive negative restaurant experiences and potential business failure?"

---

## Dataset
- Yelp Open Dataset (~4GB raw data)
- Sample of 5,000 reviews used for analysis
- Key fields:
  - `text` (review content)
  - `stars` (rating)
  - `useful` (engagement)

---

## Methodology

### 1. Data Processing
- Load large JSON dataset
- Extract relevant columns
- Convert to manageable dataset

### 2. Text Classification
Reviews are categorized into operational KPIs using keyword-based NLP:

- Service quality
- Food quality
- Price/value
- Order accuracy
- Management
- Cleanliness
- Staffing
- Ambience

### 3. KPI Metrics
Each category is evaluated using:

- Frequency (how often issues occur)
- Average rating (impact on customer satisfaction)
- Engagement (`useful` votes)
- Sentiment score (TextBlob polarity)

### 4. Failure Score
A composite score is calculated:

Failure Score =
0.4 × frequency +
0.3 × engagement +
0.2 × rating impact +
0.1 × sentiment

This ranks the most critical failure drivers.

---

## Key Findings

- Service issues are the most frequent and impactful complaints
- Food quality problems are strongly associated with low ratings
- Management-related issues correlate with systemic failures
- Price sensitivity plays a significant role in customer dissatisfaction

---

## Visualization

The project includes visualization of top failure drivers:

(Insert your chart screenshot here later)

---

## Tech Stack
- Python
- Pandas
- TextBlob (NLP)
- Matplotlib

---

## Future Improvements
- Machine learning classification model
- Topic modeling (LDA)
- Dashboard (Streamlit)
- Real-time data pipeline

---

## Author
Aaron (transitioning into Data / ML engineering)


