from django.urls import path
from . import views

urlpatterns = [
    path('', views.input_view, name='input'),  # Input form
    path('process/', views.process_input, name='process_input'),  # Process input
]
