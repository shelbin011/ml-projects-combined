from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('cpu-temp/', views.cpu_temp, name='cpu_temp'),
    path('loan-approval/', views.loan_approval, name='loan_approval'),
    path('heart-disease/', views.heart_disease, name='heart_disease'),
    path('solar-mlr/', views.solar_mlr, name='solar_mlr'),
    path('population/', views.population, name='population'),
]
