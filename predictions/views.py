from django.shortcuts import render
from django.conf import settings
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
def load_file(filename):
    path = os.path.join(settings.BASE_DIR, filename)
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    return None

def home(request):
    return render(request, 'home.html')

def cpu_temp(request):
    prediction = None
    if request.method == 'POST':
        usage = float(request.POST.get('cpu_usage'))
        model = load_file('cpu_model.pkl')
        if model:
            pred = model.predict(np.array([[usage]]))
            prediction = round(pred[0], 2)
    return render(request, 'cpu_temp.html', {'prediction': prediction})

def loan_approval(request):
    prediction = None
    if request.method == 'POST':
        inputs = [
            int(request.POST.get('no_of_dependents')),
            int(request.POST.get('education')),
            int(request.POST.get('self_employed')),
            int(request.POST.get('income_annum')),
            int(request.POST.get('loan_amount')),
            int(request.POST.get('loan_term')),
            int(request.POST.get('cibil_score')),
            int(request.POST.get('residential_assets')),
            int(request.POST.get('commercial_assets')),
            int(request.POST.get('luxury_assets')),
            int(request.POST.get('bank_assets'))
        ]
        model = load_file('loan_knn_model.pkl')
        scaler = load_file('scaler.pkl')
        if model and scaler:
            scaled_inputs = scaler.transform([inputs])
            pred = model.predict(scaled_inputs)
            prediction = 'Approved' if pred[0] == 0 else 'Rejected'
    return render(request, 'loan_approval.html', {'prediction': prediction})

def heart_disease(request):
    prediction = None
    if request.method == 'POST':
        # We supply the 6 inputs from the form, and default padding for the other 7 features
        # so it doesn't crash when using the real 13-feature notebook model.
        inputs = [
            float(request.POST.get('age')),
            float(request.POST.get('sex')),
            float(request.POST.get('cp')),
            float(request.POST.get('trestbps')),
            float(request.POST.get('chol')),
            0.0,  # fbs (fasting blood sugar): assume normal (0)
            1.0,  # restecg: assume normal (1)
            float(request.POST.get('thalach')),
            0.0,  # exang (exercise angina): assume none (0)
            1.0,  # oldpeak: assume typical median (1.0)
            2.0,  # slope: assume typical median (2.0)
            0.0,  # ca: assume typical healthy (0.0)
            2.0   # thal: assume typical normal (2.0)
        ]
        model = load_file('heart_disease_model.pkl')
        if model:
            pred = model.predict([inputs])
            prediction = 'Disease Detected' if pred[0] == 1 else 'No Disease'
    return render(request, 'heart_disease.html', {'prediction': prediction})

def solar_mlr(request):
    prediction = None
    if request.method == 'POST':
        inputs = [
            float(request.POST.get('ambient_temperature')),
            float(request.POST.get('module_temperature')),
            float(request.POST.get('irradiation'))
        ]
        model = load_file('solar_mlr_model.pkl')
        if model:
            pred = model.predict([inputs])
            prediction = round(pred[0], 2)
    return render(request, 'solar_mlr.html', {'prediction': prediction})

def population(request):
    prediction = None
    country = None
    countries = []
    csv_path = os.path.join(settings.BASE_DIR, 'world_population.csv')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        countries = sorted(df['Country/Territory'].unique().tolist())
        
        if request.method == 'POST':
            year = int(request.POST.get('year'))
            country = request.POST.get('country')
            
            country_df = df[df['Country/Territory'] == country]
            if not country_df.empty:
                # Extract years and populations as done in the notebook
                years = [1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022]
                populations = [
                    country_df["1970 Population"].values[0],
                    country_df["1980 Population"].values[0],
                    country_df["1990 Population"].values[0],
                    country_df["2000 Population"].values[0],
                    country_df["2010 Population"].values[0],
                    country_df["2015 Population"].values[0],
                    country_df["2020 Population"].values[0],
                    country_df["2022 Population"].values[0]
                ]
                
                new_df = pd.DataFrame({"Year": years, "Population": populations})
                X = new_df[["Year"]]
                y = new_df["Population"]
                
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(X)
                
                model = LinearRegression()
                model.fit(X_poly, y)
                
                # Predict for target year
                year_poly = poly.transform([[year]])
                pred = model.predict(year_poly)
                pred_val = int(pred[0])
                if pred_val < 0:
                    prediction = "0 (Out of range)"
                else:
                    prediction = f"{pred_val:,}"

    return render(request, 'population.html', {
        'prediction': prediction,
        'countries': countries,
        'selected_country': country
    })
