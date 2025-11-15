Query Optimization Engine for E-Commerce Databases (Olist Dataset)

This project implements an AI-powered Automatic Database Administrator (Auto-Admin) system that analyzes SQL queries, recommends optimal indexes, and measures performance improvements — all integrated with a FastAPI backend and ML models.

The system is built using the Olist E-commerce dataset, containing real-world Brazilian marketplace transactions (~100K+ records across multiple relational tables).

📌 Key Features
✅ Machine-Learning–Based Index Recommendation

Trained a Random Forest classifier on ~30K real + synthetic SQL queries

Multi-label index prediction using MultiLabelBinarizer

Model performance:

Accuracy: ~54%

Runtime improvement classification: ~70%

Actual SQL Runtime Speedups: up to 5–12× faster with recommended indexes

📌 Key Features
✅ Machine-Learning–Based Index Recommendation

Trained a Random Forest classifier on ~30K real + synthetic SQL queries

Multi-label index prediction using MultiLabelBinarizer

Model performance:

Accuracy: ~54%

Runtime improvement classification: ~70%

Actual SQL Runtime Speedups: up to 5–12× faster with recommended indexes

🛠️ Tech Stack
Backend

Python

FastAPI

SQLite (Olist DB replica)

Machine Learning

Scikit-learn

RandomForestClassifier

MultiLabelBinarizer

Synthetic query generator

NLP + Regex–based SQL parsing

DevOps + Tools

Pandas, Numpy

Matplotlib

Joblib

Postman for API testing