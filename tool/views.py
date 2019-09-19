# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render,render_to_response
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import matplotlib.pyplot as plt
from os import path
from PIL import Image
import tensorflow as tf
from sklearn.externals import joblib

# PremutationImportantce imports
from IPython.display import display
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance
from eli5.formatters import format_as_html
from sklearn import preprocessing
from sklearn import utils
from sklearn import metrics, svm
import shap
# Genralizing Premuation imporatnce
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
import pickle
# redirect
from django.shortcuts import redirect
import urllib
from urllib.parse import urlencode
import ast

# Takes a pickeled validation dataset and a pickeled trained model
def FeatureImportanceGenral(request):
	context = {'FeatureImportance': ''}
	if request.method== "POST":
		uploaded_dataset = request.FILES['dataset']
		uploaded_model=request.FILES['model']
		fs = FileSystemStorage()
		name_dataset = fs.save(uploaded_dataset.name,uploaded_dataset)
		name_model = fs.save(uploaded_model.name,uploaded_model)
		pth_dataset='tool/'+fs.url(name_dataset)
		pth_model='tool/'+fs.url(name_model)
		pickel_in_model=open(pth_model,"rb")
		pickel_in_Val=open(pth_dataset,"rb")
		my_model = pickle.load(pickel_in_model)
		train_X, val_X, train_y, val_y=pickle.load(pickel_in_Val)
		perm = PermutationImportance(my_model, random_state=1).fit(val_X,val_y)
		exp=eli5.explain_weights(perm, feature_names = val_X.columns.tolist())
		context['FeatureImportance']= eli5.formatters.html.format_as_html(exp, include_styles=True, force_weights=True, show=('method', 'description', 'transition_features', 'targets', 'feature_importances', 'decision_tree'), preserve_density=None, highlight_spaces=True, horizontal_layout=True, show_feature_values=True)
		display(eli5.show_weights(perm, feature_names = val_X.columns.tolist()))
		return HttpResponse(context['FeatureImportance'])
	return render(request, 'tool/PremutationImportantce.html', context)

def ShapeValuesGenral(request):
	"""This method taked a pickled dataset train_test_split and a pickled model and returns
		the shap value of the first record in the dataset.
		1.As we are using the kernal explainer we will use a summary of our dataset
	     so that we can return the results faster.By using shap.kmeans with 10 as the number of
		 means for approximations.
		 2. The use of kernal explainers approximates the result for models and is considered to be
		 slower than the other type of explainers
		 3.We used kernal explainers in order to be able to use any model during the use of shap vlaues"""
	context={'ShapeValues':''}
	if request.method=='POST':
		uploaded_dataset = request.FILES['dataset']
		uploaded_model=request.FILES['model']
		fs = FileSystemStorage()
		name_dataset = fs.save(uploaded_dataset.name,uploaded_dataset)
		name_model = fs.save(uploaded_model.name,uploaded_model)
		pth_dataset='tool/'+fs.url(name_dataset)
		pth_model='tool/'+fs.url(name_model)
		pickel_in_model=open(pth_model,"rb")
		pickel_in_Val=open(pth_dataset,"rb")
		my_model = pickle.load(pickel_in_model)
		train_X, val_X, train_y, val_y =pickle.load(pickel_in_Val)
		row_to_show=5
		# 1
		train_X_Summary=shap.kmeans(train_X, 10)
		Prediction_data= val_X.iloc[[row_to_show]]
		#2 #3
		explainer = shap.KernelExplainer(my_model.predict, train_X_Summary)
		shap_values_test = explainer.shap_values(val_X)
		shap.initjs()
		plot=shap.force_plot(explainer.expected_value, shap_values_test[row_to_show], Prediction_data,matplotlib=False)
		shap.save_html('/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/templates/tool/result_of_shap.html',plot)
		return render(request, '/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/templates/tool/result_of_shap.html',context)
	return render(request, 'tool/shap_values.html', context)

# This method is Genralized
def summaryplots(request):
	context={'url':''}
	if request.method=='POST':
		uploaded_dataset = request.FILES['dataset']
		uploaded_model=request.FILES['model']
		fs = FileSystemStorage()
		name_dataset = fs.save(uploaded_dataset.name,uploaded_dataset)
		name_model = fs.save(uploaded_model.name,uploaded_model)
		pth_dataset='tool/'+fs.url(name_dataset)
		pth_model='tool/'+fs.url(name_model)
		pickel_in_model=open(pth_model,"rb")
		pickel_in_Val=open(pth_dataset,"rb")
		my_model = pickle.load(pickel_in_model)
		train_X, val_X, train_y, val_y =pickle.load(pickel_in_Val)
		train_X_Summary=shap.kmeans(train_X, 10)
		explainer = shap.KernelExplainer(my_model.predict, train_X_Summary)
		shap_values = explainer.shap_values(train_X)
		plt.clf()
		shap.summary_plot(shap_values, train_X,show=False)
		output_path=path.join('/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/plots/image.png')
		plt.tight_layout()
		plt.savefig(output_path)
		image_data = open('/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/plots/image.png', "rb").read()
		return HttpResponse(image_data, content_type='image/png ')
	return render(request, 'tool/Summary_plots.html', context)

# This method is Genralized
def dependancycontribution(request):
	if request.method=='POST':
		uploaded_dataset = request.FILES['dataset']
		uploaded_model=request.FILES['model']
		fs = FileSystemStorage()
		name_dataset = fs.save(uploaded_dataset.name,uploaded_dataset)
		name_model = fs.save(uploaded_model.name,uploaded_model)
		pth_dataset='tool/'+fs.url(name_dataset)
		pth_model='tool/'+fs.url(name_model)
		pickel_in_model=open(pth_model,"rb")
		pickel_in_Val=open(pth_dataset,"rb")
		my_model = pickle.load(pickel_in_model)
		train_X, val_X, train_y, val_y =pickle.load(pickel_in_Val)
		feature_names=val_X.columns.tolist()
		print(feature_names)
		dropdown=', '.join(feature_names)
		print(dropdown)
		return redirect('tool-dependancycontributionhelper', feature_names=dropdown)
	return render(request, 'tool/dependency_plots.html', {'dropdown':''})

def dependancycontributionhelper(request, feature_names='dropdown'):
	dropdown=feature_names.split(', ')
	# print(dropdown)
	if request.method=='POST':
		uploaded_dataset = request.FILES['dataset']
		uploaded_model=request.FILES['model']
		fs = FileSystemStorage()
		name_dataset = fs.save(uploaded_dataset.name,uploaded_dataset)
		name_model = fs.save(uploaded_model.name,uploaded_model)
		pth_dataset='tool/'+fs.url(name_dataset)
		pth_model='tool/'+fs.url(name_model)
		pickel_in_model=open(pth_model,"rb")
		pickel_in_Val=open(pth_dataset,"rb")
		my_model = pickle.load(pickel_in_model)
		train_X, val_X, train_y, val_y =pickle.load(pickel_in_Val)
		train_X_Summary=shap.kmeans(train_X, 10)
		explainer = shap.KernelExplainer(my_model.predict, train_X_Summary)
		shap_values = explainer.shap_values(train_X)
		feature=request.POST.get('drop1')
		InteractionIndex=request.POST.get('drop2')
		plt.clf()
		shap.dependence_plot(feature, shap_values, train_X, interaction_index=InteractionIndex,show=False)
		output_path=path.join('/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/plots/image1.png')
		plt.tight_layout()
		plt.savefig(output_path)
		image_data = open('/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/plots/image1.png', "rb").read()
		return HttpResponse(image_data, content_type='image/png')
	return render(request, 'tool/dropdown.html',{'dropdown':dropdown})
