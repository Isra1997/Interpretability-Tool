from django.contrib import admin
from  .  import  views
from django.urls import path

urlpatterns = [
    path( '', views.FeatureImportanceGenral,  name='tool-home'),
    path(r'ShapeValues/', views.ShapeValuesGenral, name='tool-ShapeValues'),
    # path(r'partialplots/', views.partialplots, name='tool-partialplots'),
    path(r'summaryplots/',views.summaryplots,name='tool-summaryplots'),
    path(r'dependancycontribution/',views.dependancycontribution,name='tool-dependancycontribution'),
    path(r'dropdown/^(?P<feature_names>[\w-]+)$/',views.dependancycontributionhelper,name='tool-dependancycontributionhelper'),
]
