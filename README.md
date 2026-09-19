# Coal Production Prediction Using Machine Learning

> A machine learning application for predicting coal production from mining, ownership, coal-type, and geographical features, with an interactive Streamlit interface for real-time predictions and visualization.

---

## 📌 Project Overview

**Coal Production Prediction** is an end-to-end machine learning project developed to estimate coal production using real-world mining data.

The project applies **Random Forest Regression** to learn relationships between mining characteristics and coal production. The trained model is integrated into a **Streamlit web application**, allowing users to enter relevant mining parameters and obtain an estimated production value along with supporting visual insights.

The project demonstrates the complete machine learning workflow:

**Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction → Web Application**

---

## 🎯 Problem Statement

Coal production depends on several factors, including the type of coal, ownership of the mine, geographical location, and other mining-related characteristics.

Analyzing these factors manually can make production estimation difficult. This project aims to build a machine learning system that can use historical mining data to estimate coal production and present the results through an easy-to-use web interface.

---

## 🚀 Objectives

* Build a machine learning model for coal production prediction.
* Preprocess and transform real-world mining data.
* Handle categorical and numerical features appropriately.
* Train a regression model using Random Forest.
* Evaluate the performance of the trained model.
* Develop an interactive Streamlit application.
* Provide production-level classification and visual insights.
* Display geographical information using latitude and longitude.

---

## ✨ Key Features

### Machine Learning

* Data cleaning and preprocessing
* Feature engineering
* Categorical feature encoding
* Train-test data splitting
* Random Forest Regression
* Model evaluation
* Prediction pipeline

### Interactive Application

* Simple and user-friendly input interface
* Real-time production prediction
* Production category classification
* Prediction comparison with average production
* Geographical visualization
* Interactive charts and maps

---

## 📊 Dataset

The project uses real-world mining data containing information related to coal production and mine characteristics.

### Important Features

| Feature                 | Description                                        |
| ----------------------- | -------------------------------------------------- |
| State/UT Name           | State or Union Territory where the mine is located |
| Mine Name               | Name of the mining operation                       |
| Coal/Lignite Production | Recorded production value                          |
| Owner                   | Mine owner or operating organization               |
| Coal/Lignite            | Type of mineral                                    |
| Govt Owned/Private      | Ownership category                                 |
| TypeofMine              | Type/category of mine                              |
| Latitude                | Geographical latitude                              |
| Longitude               | Geographical longitude                             |

The dataset contains both **categorical and numerical features**, which are processed before being supplied to the machine learning model.

---

## 🤖 Machine Learning Model

### Random Forest Regression

The project uses **Random Forest Regression** as the primary prediction algorithm.

Random Forest is an ensemble machine learning algorithm that combines predictions from multiple decision trees. It is suitable for this project because it can capture non-linear relationships between mining features and production values.

### Prediction Target

The model predicts:

**Coal/Lignite Production**

The prediction is represented in **million tonnes** in the application.

---

## 🔄 Machine Learning Pipeline

```text
             Raw Mining Dataset
                     │
                     ▼
             Data Preprocessing
                     │
                     ▼
             Feature Engineering
                     │
                     ▼
             Categorical Encoding
                     │
                     ▼
              Train-Test Split
                     │
                     ▼
          Random Forest Regression
                     │
                     ▼
             Model Evaluation
                     │
                     ▼
              Trained Model
                     │
                     ▼
             Prediction Pipeline
                     │
                     ▼
          Streamlit Web Application
                     │
              ┌──────┴──────┐
              ▼             ▼
        Prediction      Visualization
```

---

## 📈 Application Output

After entering the required parameters, the application provides:

### Production Prediction

Displays the estimated coal production in **million tonnes**.

### Production Category

The predicted production is categorized as:

* **Low**
* **Medium**
* **High**

### Production Comparison

The application provides a visual comparison between the predicted production and the average production value.

### Location Visualization

The selected mine location can be visualized using its **latitude and longitude** coordinates.

---

## 🖥️ Web Application

The project uses **Streamlit** to provide an interactive interface for the trained machine learning model.

The application allows users to:

1. Enter mining-related parameters.
2. Submit the input values.
3. Generate a production prediction.
4. View the corresponding production category.
5. Analyze graphical insights.
6. View the selected geographical location.

---

## 🛠️ Tech Stack

| Category             | Technology               |
| -------------------- | ------------------------ |
| Programming Language | Python                   |
| Data Processing      | Pandas, NumPy            |
| Machine Learning     | Scikit-learn             |
| Model                | Random Forest Regression |
| Visualization        | Matplotlib, Seaborn      |
| Web Framework        | Streamlit                |
| Version Control      | Git & GitHub             |

---

## 📂 Project Structure

```text
coal_project/
│
├── data/
│   └── dataset.csv
│
├── model/
│   └── trained_model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

> File names inside `data/` and `model/` may vary depending on the current implementation.

---

## ⚙️ Installation & Setup

### Prerequisites

Make sure the following are installed:

* Python 3.x
* Git
* pip

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Navigate to the Project

```bash
cd coal_project
```

### 3. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1 — Train the Model

```bash
python src/train.py
```

This processes the dataset and trains the Random Forest regression model.

### Step 2 — Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will open in your default browser.

---

## 🧪 Model Evaluation

The model can be evaluated using regression performance metrics such as:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

These metrics help measure the difference between actual and predicted coal production values.

---

## 📊 Visualizations

The project includes visualizations to make the model results easier to understand.

Examples include:

* Production comparison charts
* Average production comparison
* Prediction-related visualizations
* Geographical map visualization

---

## 💡 What This Project Demonstrates

This project demonstrates practical experience in:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Machine learning regression
* Model evaluation
* Python-based data analytics
* Building interactive ML applications
* Integrating a trained model into a web interface
* Git and GitHub project management

---

## 🔮 Future Improvements

Potential improvements include:

* Experimenting with additional regression algorithms such as Gradient Boosting and XGBoost.
* Automated hyperparameter optimization.
* Adding historical production trend analysis.
* Integrating regularly updated mining datasets.
* Improving geographical analysis.
* Adding model performance monitoring.
* Deploying the application to a cloud platform.
* Improving the user interface and visualization dashboard.

---

## 🌐 Deployment

The application can be deployed using platforms such as:

* Streamlit Community Cloud
* AWS
* Other cloud-based hosting platforms

---

## 👨‍💻 Author

**Abhinav Yadav**

Computer Science Engineering Student
MIT World Peace University

---

## ⭐ Project

If you find this project useful, consider giving the repository a **star**.

---

## 📄 License

This project is developed for **educational and academic purposes**.
