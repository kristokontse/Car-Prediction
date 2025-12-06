# Car-Prediction (Group B1) - Kristo Kontse, Gregor Uustalu
Database we are using: https://www.kaggle.com/datasets/meruvulikith/90000-cars-data-from-1970-to-2024/data

## Motivation

The used car market is large, dynamic, and influenced by many factors such as mileage, age, brand and overall vehicle condition. Because of this complexity, it is often difficult for buyers, sellers, and dealerships to accurately estimate a fair price for a vehicle. Traditional pricing methods rely heavily on manual assessment or outdated valuation tables, which can lead to inconsistent or inaccurate results.

Machine learning offers a powerful solution by analyzing large amounts of historical data and identifying patterns that influence car prices. By using different models, it becomes possible to build a system that delivers fast, objective, and reliable price predictions.

## Our goal
The goal of this project is to develop a machine learning–based models and program which is capable of predicting used car prices. 
To achieve this, we use different models (such as Random Forest and LightGBM) to estimate the valuation of the cars.

### data_cleaning.ipynb 
Contains all the necessary code for cleaning and preprocessing the dataset.<br>

### data_analyze.ipynb
Includes visualizations and explanatory analysis performed before model training.<br>

### Models_and_price_prediction.ipynb
The main notebook of the project. It contains the machine learning models, the training and testing process, and summary graphs of the results.<br>

### CarsData.csv / CarsData_cleaned.csv
The original Kaggle dataset and the cleaned version used for analysis and modeling.<br>

### main.py
This is the program we made, where you can input various car specifications to predict the price using trained models.

## Project using
Clone this repository
   
### How to use files ipynb files
Use Jupyter Notebook to open these files. Files should have all necessary packageses installed. If something isn't installed, use Anaconda Prompt to install missing packages. Restarting Kernle in the main file (Models_and_price_prediction.ipynb) could take some time because our dataset has over 90 000 rows in it, models need to process it little longer.

### How to use main.py 
Instructions are inside the code, Open with visual studio code.

