#!/usr/bin/env python
# coding: utf-8

# ## DS/CMPSC 410 MiniProject #1
# 
# ### Spring 2021
# ### Instructor: John Yen
# ### TA: Dongkuan Xu, Rupesh Prajapati
# ### Learning Objectives
# - Be able to identify frequent 2 port sets and 3 port sets that are scanned by scanners in the Darknet dataset
# - Be able to improve the frequent port set mining algorithm by adding suitable filtering
# - Be able to improve the performance of frequent port set mining by suitable reuse of RDD, together with appropriate persist and unpersist on the reused RDD.
# 
# ### Total points: 100 
# - Exercise 1: 10 points
# - Exercise 2: 10 points
# - Exercise 3: 20 points
# - Exercise 4: 20 points
# - Exercise 5: 10 points
# - Exercise 6: 30 points (run spark-submit on a large Dataset)
# 
# ### Submit the following items for this mini project deliverable:
# - Completed Jupyter Notebook (including answers to Exercise 9.1 to 9.5; in HTML or PDF format)
# - The python file (.py) used for spark-submit
# - The output file that contains counts of 2-port sets and 3-port sets.
# - The log file of spark-submit that shows the CPU time for completing the spark-submit job.
#   
# ### Due: midnight, April 2, 2021

# In[1]:


import pyspark
import csv
import pandas as pd


# In[2]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.clustering import KMeans


# In[3]:


ss = SparkSession.builder.appName("Lab9 FrequentPortSets").getOrCreate()


# # Exercise 9.1 (10 points)
# - Complete the path below for reading "sampled_profile.csv" you downloaded from Canvas, uploaded to Lab9 folder. (5 points)
# - Fill in your Name (5 points): 

# In[4]:


Scanners_df = ss.read.csv("1.csv", header= True, inferSchema=True )


# ## We can use printSchema() to display the schema of the DataFrame Scanners_df to see whether it was inferred correctly.

# ## Part A Transfosrm the feature "ports_scanned_str" into an array of ports.
# ### The original value of the column is a string that connects all the ports scanned by a scanner. The different ports that are open by a scanner are connected by dash "-". For example, "81-161-2000" indicates the scanner has scanned three ports: port 81, port 161, and port 2000. Therefore, we want to use split to separate it into an array of ports by each scanner.  This transformation is important because it enables the identification of frequent ports scanned by scanners.

# ## The original value of the column "ports_scanned_str" 

# ## Convert the Column 'ports_scanned_str' into an Array of ports scanned by each scanner (row)

# In[7]:


Scanners_df2=Scanners_df.withColumn("Ports_Array", split(col("ports_scanned_str"), "-") )


# ## For Mining Frequent Port Sets being scanned, we only need the column ```Ports_Array```

# In[8]:


Ports_Scanned_RDD = Scanners_df2.select("Ports_Array").rdd


# ## Because each port number in the Ports_Array column for each row occurs only once, we can count the total occurance of each port number through flatMap.

# In[10]:


Ports_list_RDD = Ports_Scanned_RDD.map(lambda row: row[0] )


# In[11]:


Ports_list_RDD.persist()


# In[12]:


Ports_list2_RDD = Ports_Scanned_RDD.flatMap(lambda row: row[0] )


# In[13]:


Port_count_RDD = Ports_list2_RDD.map(lambda x: (x, 1))


# In[14]:


Port_count_total_RDD = Port_count_RDD.reduceByKey(lambda x,y: x+y, 1)


# In[16]:


Sorted_Count_Port_RDD = Port_count_total_RDD.map(lambda x: (x[1], x[0])).sortByKey( ascending = False)


# # The value of the threshold below should be identical to your choice of threshold for Exercise 9.3

# In[18]:


threshold = 5000
Filtered_Sorted_Count_Port_RDD= Sorted_Count_Port_RDD.filter(lambda x: x[0] > threshold)


# In[19]:


Top_Ports = Filtered_Sorted_Count_Port_RDD.map(lambda x: x[1]).collect()


# In[20]:


Top_1_Port_count = len(Top_Ports)


# # Exercise 9.2 (10 points)
# Compute the total number of scanners in Ports_list_RDD with the total number of scanners that scan more than one port.  What is the impact of this filter on the size of the RDD? Complete the following code to find out the answers. Then, fill the answer in the cell marked as Answer to Exercise 9.2.

# In[22]:


# Filter out those scanners that scan only 1 port
multi_Ports_list_RDD = Ports_list_RDD.filter(lambda x: len(x)>1 )


# # Answer to Exercise 9.2 
# - Original number of scanners:
# ## 227062
# - Number of scanners that scan more than one port:
# ## 73663
# - Impact of the filtering on the size of filtered scanners: 
# ## The size flitered scanners is 1/3 of the original scanners

# # Exercise 9.3 (20 points)
# - Choose a threshold (suggest a number between 500 and 1000) (5 points)
# - Complete the following code for finding 2 port sets (7 points)
# - Add suitable persist and unpersist to suitable RDD (8 points)

# In[29]:


# Initialize a Pandas DataFrame to store frequent port sets and their counts 
Freq_Port_Sets_df = pd.DataFrame( columns= ['Port Sets', 'count'])
# Initialize the index to the Freq_Port_Sets_df to 0
index = 0
# Set the threshold for Large Port Sets to be 100
threshold = 5000
multi_Ports_list_RDD.persist()
for i in range(0, Top_1_Port_count-1):
    Scanners_port_i_RDD = multi_Ports_list_RDD.filter(lambda x: Top_Ports[i] in x)
    for j in range(i+1, Top_1_Port_count-1):
        Scanners_port_i_j_RDD = Scanners_port_i_RDD.filter(lambda x:  Top_Ports[j] in x)
        two_ports_count = Scanners_port_i_j_RDD.count()
        if two_ports_count > threshold:
            Freq_Port_Sets_df.loc[index]=[ [Top_Ports[i], Top_Ports[j]], two_ports_count]
            index = index +1
    Scanners_port_i_RDD.unpersist()


# In[31]:


tri_Ports_list_RDD=multi_Ports_list_RDD.filter(lambda x: len(x)>2)


# # Exercise 9.5 (20 points)
# - Use the same threshold as Exercise 9.4 (5 points)
# - Complete the following code to find frequent 3 port sets (7 points)
# - Add persist and unpersist to suitable RDD (8 points)

# In[ ]:


# Set the threshold for Large Port Sets to be 100
threshold = 5000
tri_Ports_list_RDD.persist()
for i in range(0, Top_1_Port_count-1):
    Scanners_port_i_RDD = tri_Ports_list_RDD.filter(lambda x: Top_Ports[i] in x)
    Scanners_port_i_RDD.persist()
    for j in range(i+1, Top_1_Port_count-1):
        Scanners_port_i_j_RDD = Scanners_port_i_RDD.filter(lambda x: Top_Ports[j] in x)
        two_ports_count = Scanners_port_i_j_RDD.count()
        if two_ports_count > threshold:
            for k in range(j+1, Top_1_Port_count -1):
                Scanners_port_i_j_k_RDD = Scanners_port_i_j_RDD.filter(lambda x: Top_Ports[k] in x)
                three_ports_count = Scanners_port_i_j_k_RDD.count()
                if three_ports_count > threshold:
                    Freq_Port_Sets_df.loc[index] = [ [Top_Ports[i], Top_Ports[j], Top_Ports[k]], three_ports_count]
                    index = index + 1
    Scanners_port_i_RDD.unpersist()
tri_Ports_list_RDD.unpersist()

# In[41]:


Freq_Port_Sets_DF = ss.createDataFrame(Freq_Port_Sets_df)


# # Exercise 9.5 (10 points)
# Complete the following code to save your frequent 2 port sets and 3 port sets in an output file.

# In[44]:

import os
projectPath=os.environ.get('PROJECT')
output_path = "%s/Lab9Output"%projectPath
Freq_Port_Sets_DF.rdd.saveAsTextFile(output_path)


# # Exercise 9.6 (30 points)
# - Remove .master("local") from SparkSession statement
# - Change the input file to "/gpfs/scratch/juy1/Day_2020_profile.csv"
# - Change the output file to a different directory from the one you used in Exercise 9.4
# - Export the notebook as a .py file
# - Run spark-submit on ICDS Roar (following instructions on Canvas)
