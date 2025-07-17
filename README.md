# ✈️ Flight Price Prediction using Machine Learning

This project is developed as a capstone internship project under **Microsoft** and aims to predict the price of domestic airline tickets using machine learning regression techniques. The model helps both **travelers** and **airlines** make informed decisions based on historical flight data.

## 📌 Problem Statement

Airlines use dynamic pricing strategies influenced by multiple factors like demand, seasonality, and competition. This makes it difficult for passengers to book flights at optimal prices. The goal is to build a predictive model that uses historical data to estimate flight ticket prices accurately in advance.

---

## ✅ Proposed Solution

The solution pipeline includes:
- **Data Collection:** Dataset sourced from [Kaggle](https://www.kaggle.com) with features like airline, source, destination, times, stops, class, and more.
- **Data Preprocessing:** Handling missing values, encoding categorical variables, and normalizing numerical ones.
- **Modeling:** Implementation of machine learning models including **Random Forest** and **XGBoost Regressor**.
- **Evaluation:** Using metrics like **MAE**, **RMSE**, and **R² Score** to assess accuracy.

---

## 🛠️ Tech Stack

### Languages & Tools:
- Python 3.x
- Jupyter Notebook

### Libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`

---

## 📊 Algorithm & Deployment

- **Regression Models Used:**  
  - Random Forest Regressor  
  - XGBoost Regressor  

- **Model Training:**  
  - Data was split into train/test sets  
  - Hyperparameter tuning  
  - Cross-validation  

- **Input Features:**  
  - Airline, Source, Destination, Departure Time, Arrival Time, Stops, Class, Duration, Days Left

---

## 📈 Results

- **R² Score:** `0.98`  
- **Model Accuracy:** `97.5%`  
- **Runtime:** `~801 seconds`

### Visuals:
- Actual vs. Predicted Price Plot  
- Feature Importance Graph

---

## 🧠 Conclusion

The project demonstrates high performance in predicting domestic flight prices using structured data and regression techniques. It assists:
- **Travelers** in booking at the best prices
- **Airlines** in optimizing dynamic pricing strategies

---

## 🔮 Future Scope

- Incorporate external data (e.g., holiday calendars, seasonal trends)
- Use Deep Learning models (LSTM, RNN) for sequential data
- Deploy as a **web/mobile application**
- Expand to **international routes** and multi-leg trips
- Integrate real-time data for dynamic updates

---

## 🔗 References

- [Flight Price Dataset - Kaggle](https://www.kaggle.com)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [GitHub Project Repo](https://github.com/mridul2139/FLIGHT_PRICE_ML)

---

## 🙋‍♂️ Author

**Mridul Makkar**  
Department: IIoT, GGSIPU EDC  
📧 Email: mastermridul2005@gmail.com  
🎓 AICTE Student ID: STU66C74409079BA1724335113  

---

> _"Machine learning has the power to decode pricing strategies and bring transparency to travel planning."_  
