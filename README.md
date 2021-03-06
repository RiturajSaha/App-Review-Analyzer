# App-Review-Analyzer

Android App Industry is growing significantly and this increases competition among those who build applicationsinthedeveloping field. This is a kind of ultimatum or warning to each individual developer as a larger number of competitions would increase the likelihood of weakening their app's play store value. App creators, therefore consequently should understand their customers' needs to make sure their applications get a better response as soon as it is deployed in the playstore.

In this respository, an App Review Analyzer application is created having a GUI based frontend to provide a simple interface to user having only one button to perform all the tasks and a machine learning backend to provie results with highest possible accuracy. Here an NLP based classification model is used to predict the sentiment of the app reviews and then various statistics and insights are generated based on those sentiments by comparing them to other app attributes like ratings, total reviews, etc which gives a clear idea of the accomplishment of the app in Google play store.

Below are the screenshots of the application:
<p align="center"><img src="https://github.com/RiturajSaha/App-Review-Analyzer/blob/main/Screenshots/1.png" height=500 width="800"></p>
<p align="center"><img src="https://github.com/RiturajSaha/App-Review-Analyzer/blob/main/Screenshots/2.png" height=500 width="800"></p>
<p align="center"><img src="https://github.com/RiturajSaha/App-Review-Analyzer/blob/main/Screenshots/4.png" height=500 width="800"></p>

The Natural Language Processing model was creted by training on a dataset obtained from [Google Play Store Apps](https://www.kaggle.com/lava18/google-play-store-apps), then cleaning and processing it, applying the Bag of Words model to it and finally using various classifcation models, in order to achieve the highest accuracy. The  model was trained in Google Colaboratory and the application was developed in Spyder,

The libraries used to build this application are:  
Numpy, Pandas, Regex, NLTK, Pickle, Scit-learn, Pillow, Matplotlib, Google-Play-Scrapper, Googlesearch, Shutil, OS, and tabulate.

Out of all the classification models applied to the dataset, Random Forest Classifier yields the highest accuracy of 90.11% and r2_scoew of 0.603, below are the other attributes of the model:  

<p align="center"><img src="https://github.com/RiturajSaha/App-Review-Analyzer/blob/main/Screenshots/acc.png" height=500 width="800"></p>

accuracy_score is the percentage of the success of a model to predcit the independent attribute and r2_score is a statistical measure that represents the goodness of fit of a regression model. The ideal value for r2_score is 1, its range is from -1 to 1. Some other methods to determine the success of a classification model are mean_squared_error, mean_absolute_error, confusion_matrix, calssification_report. 

The model is deployed in the backend of the appliccation to classify the reviews of apps into Positive, Neutral and, Neagtive and then generate useful insights from it.
