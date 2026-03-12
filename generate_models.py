import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

print("Generating dummy models...")

# 1. CPU Temp (Simple Linear Regression)
# Features: CPU Usage. Target: Temp
cpu_model = LinearRegression()
cpu_model.fit(np.array([[10], [50], [90]]), np.array([35, 62, 92]))
pickle.dump(cpu_model, open('cpu_model.pkl', 'wb'))

# 2. Loan Approval (KNN)
# Features: 11 inputs
# [no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets, commercial_assets, luxury_assets, bank_assets]
X_loan = np.array([
    [2, 0, 0, 9600000, 29900000, 12, 778, 2400000, 17600000, 22700000, 8000000],
    [0, 1, 1, 2000000, 5000000, 24, 500, 1000000, 2000000, 3000000, 1000000]
])
y_loan = np.array([0, 1]) # 0 = Approved, 1 = Rejected
scaler = StandardScaler()
X_loan_scaled = scaler.fit_transform(X_loan)
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_loan_scaled, y_loan)
pickle.dump(knn_model, open('loan_knn_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# 3. Heart Disease (Logistic Regression)
# Features: 13 inputs
# Supplying 4 dummy rows (2 healthy [0], 2 disease [1])
X_heart = np.array([
    [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3], # 0
    [45, 0, 1, 110, 150, 0, 0, 140, 0, 0.5, 1, 0, 2], # 1
    [60, 1, 2, 140, 250, 1, 1, 130, 1, 2.0, 1, 1, 3], # 0
    [35, 0, 0, 100, 120, 0, 1, 180, 0, 0.0, 2, 0, 2], # 1
])
y_heart = np.array([0, 1, 0, 1])
heart_model = LogisticRegression(max_iter=1000)
heart_model.fit(X_heart, y_heart)
pickle.dump(heart_model, open('heart_disease_model.pkl', 'wb'))

# 4. Solar MLR (Multiple Linear Regression)
# Features: 3 inputs (AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION)
X_solar = np.array([
    [25, 30, 0.8],
    [30, 45, 1.2]
])
y_solar = np.array([300, 500])
solar_model = LinearRegression()
solar_model.fit(X_solar, y_solar)
pickle.dump(solar_model, open('solar_mlr_model.pkl', 'wb'))

# 5. Population Prediction (Polynomial Regression)
# Features: Year -> Poly 3
X_pop = np.array([[1970], [2000], [2022]])
y_pop = np.array([555000000, 1050000000, 1410000000])
poly = PolynomialFeatures(degree=3)
X_pop_poly = poly.fit_transform(X_pop)
poly_model = LinearRegression()
poly_model.fit(X_pop_poly, y_pop)
pickle.dump(poly_model, open('pop_model.pkl', 'wb'))
pickle.dump(poly, open('poly_transform.pkl', 'wb'))

print("Models generated successfully!")
