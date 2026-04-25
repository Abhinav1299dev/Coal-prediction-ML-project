#  Coal Production Prediction using Machine Learning

## 📌 Overview

This project predicts **coal production levels** using a Machine Learning model based on real-world mining data.
It leverages **Random Forest Regression** and provides an **interactive web interface** built with Streamlit.

The system allows users to input mining-related parameters and instantly get:

* 📊 Estimated coal production (in million tonnes)
* 📈 Production level (Low / Medium / High)
* 📉 Visual comparison with average production
* 🗺️ Location-based visualization on map

---

## 🚀 Key Features

* 🔹 Data preprocessing & feature engineering
* 🔹 Handling categorical data using encoding
* 🔹 Model training using Random Forest
* 🔹 Model evaluation with visualizations
* 🔹 Interactive Streamlit web application
* 🔹 Real-time prediction system
* 🔹 Graph + Map visualization for better insights

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Web App:** Streamlit

---

## 📊 Input Parameters

The model uses the following key inputs:

* State/UT Name
* Coal Type (Coal / Lignite)
* Ownership (Govt / Private)
* Latitude & Longitude

---

## 📈 Output

The system provides:

* ✅ Estimated coal production (in million tonnes)
* ✅ Production category (Low / Medium / High)
* ✅ Graph comparison with average production
* ✅ Map visualization of selected location

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the model

```bash
python src/train.py
```

### 3️⃣ Run the web application

```bash
python -m streamlit run app.py
```

---

## 📂 Project Structure

```
coal_project/
│
├── data/
├── model/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 💡 Key Highlights

* ✔️ End-to-end ML pipeline (data → model → deployment)
* ✔️ Clean and modular code structure
* ✔️ Real-world problem solving
* ✔️ User-friendly interactive UI

---

## 🚀 Future Improvements

* 🔸 Deploy the app on cloud (Streamlit Cloud / AWS)
* 🔸 Add real-time data integration
* 🔸 Improve model accuracy with advanced algorithms
* 🔸 Add historical trend analysis

---

## 👨‍💻 Author

**Abhinav Yadav**

---

⭐ If you like this project, consider giving it a star!
