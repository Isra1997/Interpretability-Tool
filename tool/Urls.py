from django.contrib import admin
from  .  import  views
from django.urls import path

urlpatterns = [
    path( '', views.featureimportance ,  name='tool-home'),
    path(r'ShapeValues/', views.ShapeValues, name='tool-ShapeValues'),
    path(r'summaryplots/',views.summaryplots,name='tool-summaryplots'),
    path(r'loaddropdown/',views.loaddropdown,name='tool-loaddropdown'),
    path(r'dependancycontribution/',views.dependancycontribution,name='tool-dependancycontribution'),
]

