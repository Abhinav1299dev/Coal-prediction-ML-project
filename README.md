# Coal Production Prediction Using Machine Learning

## Overview

**Coal Production Prediction** is a machine learning project that predicts coal production levels using real-world mining data.

The project uses a **Random Forest Regression** model to estimate coal production based on mining-related and geographical parameters. An interactive **Streamlit web application** allows users to enter input values and receive real-time predictions along with visual insights.

### Key Capabilities

* Predict estimated coal production in **million tonnes**
* Classify production into **Low, Medium, or High**
* Compare predicted production with average production
* Visualize the selected mining location on an interactive map
* Provide an end-to-end workflow from data preprocessing to deployment

---

## Key Features

* **Data Preprocessing** – Cleaning and preparing real-world mining data
* **Feature Engineering** – Transforming raw data into model-ready features
* **Categorical Encoding** – Converting categorical mining attributes into numerical representations
* **Random Forest Regression** – Training a machine learning model for production prediction
* **Model Evaluation** – Evaluating model performance using appropriate metrics and visualizations
* **Interactive Web Application** – User-friendly Streamlit interface for predictions
* **Data Visualization** – Graphical comparison of predicted and average production
* **Location Visualization** – Map-based representation using latitude and longitude

---

## Technology Stack

| Category             | Technologies        |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Data Processing      | Pandas, NumPy       |
| Machine Learning     | Scikit-learn        |
| Data Visualization   | Matplotlib, Seaborn |
| Web Application      | Streamlit           |
| Version Control      | Git, GitHub         |

---

## Input Parameters

The model uses the following parameters to generate predictions:

* **State/UT Name**
* **Coal Type** – Coal / Lignite
* **Ownership** – Government / Private
* **Latitude**
* **Longitude**

---

## Output

Based on the provided inputs, the application generates:

* **Estimated Coal Production** in million tonnes
* **Production Category** – Low / Medium / High
* **Production Comparison** against average production
* **Location Visualization** based on the selected latitude and longitude

---

## Machine Learning Workflow

The project follows an end-to-end machine learning pipeline:

```text
Raw Mining Data
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Categorical Encoding
      ↓
Train-Test Split
      ↓
Random Forest Regression
      ↓
Model Evaluation
      ↓
Prediction
      ↓
Streamlit Web Application
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd coal_project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python src/train.py
```

### 4. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will open in your browser, where you can enter mining parameters and generate predictions.

---

## Project Structure

```text
coal_project/
│
├── data/
│
├── model/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Project Highlights

* End-to-end machine learning implementation
* Real-world mining and geological data
* Modular and maintainable project structure
* Regression-based production forecasting
* Interactive prediction interface using Streamlit
* Graphical and geographical data visualization
* Integration of machine learning with a deployable web application

---

## Future Improvements

* Deploy the application using **Streamlit Cloud or AWS**
* Integrate updated or real-time mining datasets
* Experiment with additional regression algorithms
* Perform further hyperparameter optimization
* Add historical production trend analysis
* Improve visualization and analytical capabilities
* Add model monitoring and performance tracking

---

## Author

**Abhinav Yadav**
Computer Science Engineering Student

---

## License

This project is intended for educational and academic purposes.
