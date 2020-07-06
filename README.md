# Predicting bike demand using linear regression

Bike sharing demand
---------------------------
The goal of this project is to predict bike sharing demand at any given hour. For that we build a simple machine learning model based on ordinary least squares regression. Find more information on the Kaggle competition and the data provided [here](https://www.kaggle.com/c/bike-sharing-demand/overview).

View the project in the following order:
1. data_exploration.ipynb
  - follow the insights extracted through data exploration
2. ols_linear_regression.ipynb
  - follow the steps taken to build and improve the model
3. ols_linear_regression_assumptions.ipynb
  - verify linear regression assumptions

View the analyses directly by clicking the jupyter notebook file in the GitHub repository. GitHub renders the files for immediate viewing. The following installation steps are only necessary if the code is to be used.

Installation
---------------------------

### Download the data

* Clone this repo to your computer.
* Get into the folder using `cd Predicting-bike-demand-using-linear-regression`.
* Switch into the `Data` directory using `cd Data`.
* Download the data files from Kaggle.  
    * You can find the data [here](https://www.kaggle.com/c/bike-sharing-demand/data).
* Extract all of the `.zip` files you downloaded.
    * copy "train.csv" and "test.csv" files into the folder `Predicting-bike-demand-using-linear-regression/Data`
    * remove zip files from `Downloads`.


### Install the requirements

* Install the requirements using `pip install -r requirements.txt`.
    * Make sure you use Python 3.
    * You may want to use a virtual environment for this.

Usage
-----------------------
* open any notebook using `jupyter-lab notebook.ipynb`.
* Run notebook by selecting Kernel -> Restart Kernel and Run All Cells
