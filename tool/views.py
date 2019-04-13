# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render,render_to_response
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import matplotlib.pyplot as plt
from os import path
from PIL import Image
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
# imports for partial plots
# from sklearn.tree import DecisionTreeClassifier
# from matplotlib import pyplot as plt
# from pdpbox import pdp, get_dataset, info_plots



# Create your views here.

# 1. make sure to display a message if a target or a dataset is not set
# 2. genralize for the all datasets

def  featureimportance(request):
	context = {'FeatureImportance': ''}
	if request.method== "POST":
		uploaded_dataset = request.FILES['dataset']
		fs = FileSystemStorage()
		name = fs.save(uploaded_dataset.name,uploaded_dataset)
		pth='tool/'+fs.url(name)
		read_file = pd.read_csv(pth)
		data = read_file.iloc[0:129]
		y = (data[request.POST.get('target')])  # Convert from string "Yes"/"No" to binary
		feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
		X = data[feature_names]
		train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1)
		lab_enc = preprocessing.LabelEncoder()
		training_scores_encoded = lab_enc.fit_transform(train_y)
		preiction_scores_encoded = lab_enc.fit_transform(val_y)
		my_model = RandomForestClassifier(random_state=0).fit(train_X, training_scores_encoded)
		perm = PermutationImportance(my_model, random_state=1)
		perm.fit(val_X,preiction_scores_encoded)
		exp=eli5.explain_weights(perm, feature_names = val_X.columns.tolist())
		context['FeatureImportance']= eli5.formatters.html.format_as_html(exp, include_styles=True, force_weights=True, show=('method', 'description', 'transition_features', 'targets', 'feature_importances', 'decision_tree'), preserve_density=None, highlight_spaces=True, horizontal_layout=True, show_feature_values=True)
		display(eli5.show_weights(perm, feature_names = val_X.columns.tolist()))
		return HttpResponse(context['FeatureImportance'])
	return render(request, 'tool/PremutationImportantce.html', context)

# 1.genralize for all datasets 
# 2.make sure to display a message if a target or a dataset is not set
# 3.taka a feature to plot

def partialplots(request):
	context ={''}
	if request.method=='POST':
		uploaded_dataset = request.FILES['dataset']
		fs = FileSystemStorage()
		name = fs.save(uploaded_dataset.name,uploaded_dataset)
		pth='tool/'+fs.url(name)
		data = pd.read_csv(pth)
		y = data[request.POST.get('target')] # Convert from string "Yes"/"No" to binary
		feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
		X = data[feature_names]
		train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
		tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
		# Create the data that we will plot
		pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=feature_names, feature='Goal Scored')
		# plot it
		pdp.pdp_plot(pdp_goals, 'Goal Scored')
		plt.show()

def ShapeValues(request):
	context={'ShapeValues':''}
	if request.method=='POST':
		uploaded_dataset = request.FILES['dataset']
		fs = FileSystemStorage()
		name = fs.save(uploaded_dataset.name,uploaded_dataset)
		pth='tool/'+fs.url(name)
		data = pd.read_csv(pth)
		y=data[request.POST.get('target')]
		feature_names = [i for i in data.columns if i != request.POST.get('target') and (data[i].dtype in [np.int64, np.int64, bool]) ]
		X=data[feature_names]
		train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
		my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
		perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
		eli5.show_weights(perm, feature_names = val_X.columns.tolist())
		data_for_prediction = val_X.iloc[0:]  # use 1 row of data here. Could use multiple rows if desired
		# Create object that can calculate shap values
		explainer = shap.TreeExplainer(my_model)
		shap_values = explainer.shap_values(data_for_prediction)
		shap.initjs()
		temp=shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction,matplotlib=False)
		shap.save_html('/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/templates/tool/result_of_shap.html',temp)
		return render(request, '/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/templates/tool/result_of_shap.html',context)
	return render(request, 'tool/shap_values.html', context)


def summaryplots(request):
	context={'url':''}
	if request.method=='POST':
		uploaded_dataset = request.FILES['dataset']
		fs = FileSystemStorage()
		name = fs.save(uploaded_dataset.name,uploaded_dataset)
		pth='tool/'+fs.url(name)
		read_file = pd.read_csv(pth)
		data = read_file.iloc[0:150]
		y=data[request.POST.get('target')]
		feature_names = [i for i in data.columns if i != request.POST.get('target') and (data[i].dtype in [np.int64] or data[i].dtype in [bool]) ]
		X=data[feature_names]
		train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
		my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
		explainer = shap.TreeExplainer(my_model)
		shap_values = explainer.shap_values(val_X)
		shap.summary_plot(shap_values[1], val_X,show=False)
		output_path=path.join('/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/plots/image.png')
		plt.savefig(output_path)
		image_data = open('/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/plots/image.png', "rb").read()
		return HttpResponse(image_data, content_type='image/png ')
	return render(request, 'tool/Summary_plots.html', context)

def loaddropdown(request):
	context={'dropdown':''}
	if request.method=='POST':
		uploaded_dataset = request.FILES['dataset']
		fs = FileSystemStorage()
		name = fs.save(uploaded_dataset.name,uploaded_dataset)
		pth='tool/'+fs.url(name)
		data = pd.read_csv(pth)
		feature_names = [i for i in data.columns if i != request.POST.get('target') and (data[i].dtype in [np.int64] or data[i].dtype in [bool]) ]
		# context['dropdown']=feature_names
		print(feature_names)
		return render(request, 'tool/dropdown.html', {'dropdown':feature_names})
	if request.method=='dep':
		uploaded_dataset = request.FILES['dataset']
		fs = FileSystemStorage()
		name = fs.save(uploaded_dataset.name,uploaded_dataset)
		pth='tool/'+fs.url(name)
		read_file = pd.read_csv(pth)
		data = read_file.iloc[0:150]
		y=data[request.dep.get('target')]
		feature_names = [i for i in data.columns if i != request.dep.get('target') and (data[i].dtype in [np.int64] or data[i].dtype in [bool]) ]
		X=data[feature_names]
		train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
		my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
		explainer = shap.TreeExplainer(my_model)
		shap_values = explainer.shap_values(val_X)
		shap.dependence_plot(request.dep.get('drop1'), shap_values[1], X, interaction_index=request.dep.get('drop2'))
		output_path=path.join('/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/plots/image1.png')
		plt.savefig(output_path)
		image_data = open('/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/plots/image1.png', "rb").read()
		return HttpResponse(image_data, content_type='image/png')
	return render(request, 'tool/PremutationImportantce.html', context)

def dependancycontribution(request):
	context={'dropdown':''}
	if request.method=='POST':
		uploaded_dataset = request.FILES['dataset']
		fs = FileSystemStorage()
		name = fs.save(uploaded_dataset.name,uploaded_dataset)
		pth='tool/'+fs.url(name)
		read_file = pd.read_csv(pth)
		data = read_file.iloc[0:150]
		y=data[request.POST.get('target')]
		feature_names = [i for i in data.columns if i != request.POST.get('target') and (data[i].dtype in [np.int64] or data[i].dtype in [bool]) ]
		X=data[feature_names]
		train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
		my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
		explainer = shap.TreeExplainer(my_model)
		shap_values = explainer.shap_values(val_X)
		shap.dependence_plot('Ball Possession %', shap_values[1], val_X, interaction_index='Goal Scored',show=False)
		output_path=path.join('/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/plots/image1.png')
		plt.savefig(output_path)
		image_data = open('/Users/israragheb/Desktop/Interpertabilitiytool/Mytool/tool/plots/image1.png', "rb").read()
		return HttpResponse(image_data, content_type='image/png')
	return render(request, 'tool/dependency_plots.html', context)
