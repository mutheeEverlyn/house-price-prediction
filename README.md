# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project involves predicting the **median housing value (MedHouseVal)** for different regions in California based on features like:

- Median Income (`MedInc`)
- House Age (`HouseAge`)
- Average Number of Rooms (`AveRooms`)
- Average Bedrooms (`AveBedrms`)
- Population (`Population`)
- Average Occupancy (`AveOccup`)
- Geographic Coordinates (`Latitude`, `Longitude`)

The goal is to use machine learning algorithms to build a predictive model using libraries like **Scikit-learn**, **Pandas**, and **NumPy** — all within an **Anaconda** environment.

---

## ✨ Key Features

✅ **Data Preprocessing**  
✔️ Cleaning and transforming the data  
✔️ Handling missing values  
✔️ Scaling features using `StandardScaler`

✅ **Model Training**  
✔️ Trains multiple models:
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  

✅ **Model Evaluation**  
✔️ Uses the following metrics:
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- R² Score

✅ **Prediction**  
✔️ Predicts housing value for new unseen input data  
✔️ Automatically scales input before prediction

---

## 🛠 Technologies Used

- [Anaconda](https://www.anaconda.com/): Python/R distribution for scientific computing  
- **Python 3.x**  
- **Scikit-learn** – Model training and evaluation  
- **Pandas** – Data loading and manipulation  
- **NumPy** – Numerical operations  
- **Matplotlib** / **Seaborn** – Data visualization  

---

## 🚀 Installation Instructions

1. **Install Anaconda**  
   Download and install from [here](https://www.anaconda.com/products/individual).

2. **Clone the Repository**  
   ```bash
   git clone <repository_url>
   cd <project_directory>
   
⚠️ Error Handling
Includes exception handling for:

ValueError: Raised during prediction or transformation errors

KeyError: Raised for missing/incorrect columns

Generic Exception: Catches all unexpected runtime errors

✅ Conclusion
This project showcases how machine learning can be used to accurately predict housing prices. By comparing multiple models and evaluating their performance, the most suitable algorithm can be chosen for deployment. This model can be applied in:

Real estate analytics

Market valuation systems

Investment decision support tools