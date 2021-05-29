#!/usr/bin/env python
# coding: utf-8

# ## DS/CMPSC 410 Sparing 2021
# ## Instructor: Professor John Yen
# ## TA: Rupesh Prajapati and Dongkuan Xu
# ## Lab 6: Movie Recommendations Using Alternative Least Square
# ## The goals of this lab are for you to be able to
# ### - Use Alternating Least Squares (ALS) for recommending movies based on reviews of users
# ### - Be able to understand the raionale for splitting data into training, validation, and testing.
# ### - Be able to tune hyper-parameters of the ALS model in a systematic way.
# ### - Be able to store the results of evaluating hyper-parameters
# ### - Be able to select best hyper-parameters and evaluate the chosen model with testing data
# ### - Be able to improve the efficiency through persist or cache
# ### - Be able to develop and debug in ICDS Jupyter Lab
# ### - Be able to run Spark-submit (Cluster Mode) in Bridges2 for large movie reviews dataset
# 
# ## Exercises: 
# - Exercise 1: 5 points
# - Exercise 2: 5 points
# - Exercise 3: 5 points
# - Exercise 4: 10 points
# - Exercise 5: 5 points
# - Exercise 6: 15 points
# - Exercise 7: 30 points
# ## Total Points: 75 points
# 
# # Due: midnight, February 28, 2021

# # Submission of Lab 6
# - 1. Completed Jupyter Notebook of Lab 6 (Lab6A.ipynb) for small movie review datasets (movies_2.csv, ratings_2.csv).
# - 2. Lab6B.py (for spark-submit on Bridges2, incorporated all improvements from Exercise 6, processes large movie reviews)
# - 3. The output file that has the best hyperparameter setting for the large movie ratings files.  
# - 4. The log file of spark-submit on Lab6B.py 
# - 5. A Word File that discusses (1) your answer to Exercise 6, and (2) your results of Exercise 7, including screen shots of your run-time information in the log file.

# ## The first thing we need to do in each Jupyter Notebook running pyspark is to import pyspark first.

# In[37]:


import pyspark


# ### Once we import pyspark, we need to import "SparkContext".  Every spark program needs a SparkContext object
# ### In order to use Spark SQL on DataFrames, we also need to import SparkSession from PySpark.SQL

# In[38]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row
from pyspark.mllib.recommendation import ALS
# from pyspark.ml import Pipeline
# from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, IndexToString
# from pyspark.ml.clustering import KMeans


# ## We then create a Spark Session variable (rather than Spark Context) in order to use DataFrame. 
# - Note: We temporarily use "local" as the parameter for master in this notebook so that we can test it in ICDS Roar.  However, we need to change "local" to "Yarn" before we submit it to XSEDE to run in cluster mode.

# In[39]:


ss=SparkSession.builder.appName("lab6").getOrCreate()


# ## Exercise 1 (5 points) (a) Add your name below AND (b) replace the path below with the path of your home directory.
# ## Answer for Exercise 1
# - a: Your Name: 
# ### Kangdong Yuan

# ## Exercise 2 (5 points) Modify the pathnames so that you can read the input CSV files (movies_2 and ratings_2 from ICDS Jupyter Lab) from the correct location.

# In[40]:


movies_DF = ss.read.csv("movies_2.csv", header=True, inferSchema=True)


# In[26]:


# movies_DF.printSchema()


# In[41]:


ratings_DF = ss.read.csv("ratings_2.csv", header=True, inferSchema=True)


# In[28]:


# ratings_DF.printSchema()


# In[42]:


ratings2_DF = ratings_DF.select("UserID","MovieID","Rating")


# In[30]:


# ratings2_DF.first()


# In[43]:


ratings2_RDD = ratings2_DF.rdd


# # 6.1 Split Data into Three Sets: Training Data, Evaluatiion Data, and Testing Data

# In[44]:


training_RDD, validation_RDD, test_RDD = ratings2_RDD.randomSplit([3, 1, 1], 137)


# ## Prepare input (UserID, MovieID) for validation and for testing

# In[45]:


import pandas as pd
import numpy as np
import math


# In[46]:


validation_input_RDD = validation_RDD.map(lambda x: (x[0], x[1])) 
testing_input_RDD = test_RDD.map(lambda x: (x[0], x[1]) )


# # 6.2 Iterate through all possible combination of a set of values for three hyperparameters for ALS Recommendation Model:
# - rank (k)
# - regularization
# - iterations 
# ## Each hyperparameter value combination is used to construct an ALS recommendation model using training data, but evaluate using Evaluation Data
# ## The evaluation results are saved in a Pandas DataFrame 
# ``
# hyperparams_eval_df
# ``
# ## The best hyperprameter value combination is stored in 4 variables
# ``
# best_k, best_regularization, best_iterations, and lowest_validation_error
# ``

# # improve the performance by use presist() method

# In[52]:


training_RDD.persist()
validation_input_RDD.persist()
validation_RDD.persist()


# # Exercise 3 (15 points) Complete the code below to iterate through a set of hyperparameters to create and evaluate ALS recommendation models.

# In[53]:


## Initialize a Pandas DataFrame to store evaluation results of all combination of hyper-parameter settings
hyperparams_eval_df = pd.DataFrame( columns = ['k', 'regularization', 'iterations', 'validation RMS', 'testing RMS'] )
# initialize index to the hyperparam_eval_df to 0
index =0 
# initialize lowest_error
lowest_validation_error = float('inf')
# Set up the possible hyperparameter values to be evaluated
iterations_list = [10, 15, 20]
regularization_list = [0.01, 0.05, 0.1]
rank_list = [4, 8, 12]
for k in rank_list:
    for regularization in regularization_list:
        for iterations in iterations_list:
            seed = 37
            # Construct a recommendation model using a set of hyper-parameter values and training data
            model = ALS.train(training_RDD, k, seed=seed, iterations=iterations, lambda_=regularization)
            # Evaluate the model using evalution data
            # map the output into ( (userID, movieID), rating ) so that we can join with actual evaluation data
            # using (userID, movieID) as keys.
            validation_prediction_RDD= model.predictAll(validation_input_RDD).map(lambda x: ( (x[0], x[1]), x[2] )   )
            validation_evaluation_RDD = validation_RDD.map(lambda y: ( (y[0], y[1]), y[2]) ).join(validation_prediction_RDD)
            # Calculate RMS error between the actual rating and predicted rating for (userID, movieID) pairs in validation dataset
            error = math.sqrt(validation_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())
            # Save the error as a row in a pandas DataFrame
            hyperparams_eval_df.loc[index] = [k, regularization, iterations, error, float('inf')]
            index = index + 1
            # Check whether the current error is the lowest
            if error < lowest_validation_error:
                best_k = k
                best_regularization = regularization
                best_iterations = iterations
                best_index = index
                lowest_validation_error = error

print('The best rank k is ', best_k, ', regularization = ', best_regularization, ', iterations = ',      best_iterations, '. Validation Error =', lowest_validation_error)


# In[54]:


# print(hyperparams_eval_df)


# # Use Testing Data to Evaluate the Model built using the Best Hyperparameters                

# # 6.4 Evaluate the best hyperparameter combination using testing data

# # Exercise 4 (10 points)
# Complete the code below to evaluate the best hyperparameter combinations using testing data.

# In[55]:


seed = 37
model = ALS.train(training_RDD, 4, seed=seed, iterations=20, lambda_=0.1)
testing_prediction_RDD=model.predictAll(testing_input_RDD).map(lambda x: ((x[0], x[1]), x[2]))
testing_evaluation_RDD= test_RDD.map(lambda x: ((x[0], x[1]), x[2])).join(testing_prediction_RDD)
testing_error = math.sqrt(testing_evaluation_RDD.map(lambda x: (x[1][0] - x[1][1])**2).mean())
print('The Testing Error for rank k =', best_k, ' regularization = ', best_regularization, ', iterations = ',       best_iterations, ' is : ', testing_error)


# In[17]:


# print(best_index)


# In[18]:


# Store the Testing RMS in the DataFrame
hyperparams_eval_df.loc[best_index]=[best_k, best_regularization, best_iterations, lowest_validation_error, testing_error]


# In[56]:


schema3= StructType([ StructField("k", FloatType(), True),                       StructField("regularization", FloatType(), True ),                       StructField("iterations", FloatType(), True),                       StructField("Validation RMS", FloatType(), True),                       StructField("Testing RMS", FloatType(), True)                     ])


# ## Convert the pandas DataFrame that stores validation errors of all hyperparameters and the testing error for the best model to Spark DataFrame
# 

# In[57]:


HyperParams_Tuning_DF = ss.createDataFrame(hyperparams_eval_df, schema3)


# # Exercise 5 (5 points)
# Modify the output path so that your output results can be saved in a directory.

# In[1]:


import os
projectPath=os.environ.get('PROJECT')
output_path = "%s/Lab6output"%projectPath
HyperParams_Tuning_DF.rdd.saveAsTextFile(output_path)


# In[58]:


#output_path = "/storage/home/kky5082/ds410/lab6/Lab6_Output"
#HyperParams_Tuning_DF.rdd.saveAsTextFile(output_path)


# # Exercise 6 (15 points)
# Modify the code above to improve its performance for Big Data.  Describe briefly here what modificationas you made and the rationale of each of your modifications.  Include this in item 5 of Lab6 submission

# ### In the train and tunning hyperparameters part, there are three for loops to test all the possible hyperparameters set. So, it reuse some rdd many times. When program reuse the rdd, it has to computate it again. But I add presist() function to store those rdd in memory or disk, the computer don't need to computate again when program reuse it. 

# # Exercise 7 (30 points)
# Save a duplicate copy of this notebook, name it Lab6B. 
# - (1) Make similar modifications to Lab6 to prepare the notebook for processing the large movie reviews in Bridges2:
# * Modify the paths for the two large input files 
# * Modify the path for the output file
# - (2) Export the notebook as Executable scripts (.py) file
# - (3) Use scp to copy the file to Bridges2
# - (4) Run (and time) spark-submit of Lab6B.py on Bridges2

# # Submission of Lab 6
# - 1. Completed Jupyter Notebook of Lab 6 (Lab6A.ipynb) for small movie review datasets (movies_2.csv, ratings_2.csv).
# - 2. Lab6B.py (for spark-submit on Bridges2, incorporated all improvements from Exercise 6, processes large movie reviews)
# - 3. The output file that has the best hyperparameter setting for the large movie ratings files.  
# - 4. The log file of spark-submit on Lab6B.py 
# - 5. A Word File that discusses (1) your answer to Exercise 6, and (2) your results of Exercise 7, including screen shots of your run-time information in the log file.

# In[ ]:




