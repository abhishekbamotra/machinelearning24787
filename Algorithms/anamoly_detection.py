import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from utils import *
from distance_functions import*
np.random.seed(0)

# READING DATA FROM THE DATASET TO PANDAS DATA FRAME
x=pd.read_excel('DSL-StrongPasswordData.xls')

# REMOVING TWO UNWANTED COLUMN FROM THE DATA SET
del x['sessionIndex']
del x['rep']

# EXTRACTING LIST OF USER NAMES FROM THE DATASET (IN THIS CASE 51)
x1=x.groupby('subject')
user_names=x1.subject.unique()



# CREATING A DICTIONARY CALLED 'all_user_data' WHERE DATA IS STORED AS FOLLOWS
# {KEY : VALUE}
# {USER_i : PANDAS DATAFRAME FOR USER i}
# EXAMPLE: {"user1" : data frame for user 1}
user_all_data = {}



# CREATING A DICTIONARY CALLED 'usernames' WHERE ORIGINAL USER NAMES ARE MAPPED TO NEW USER NAMES
# {NEW USER NAME : ORIGINAL USER NAME}
# EXAMPLE: {"user1" : "s002"}
usernames = {}

for i, user in enumerate(user_names):

	# CREATING NAME OF USER
	name = "user"+str(i+1)

	# STORING USER NAME AND DATA FRAMES TO DICTIONARY
	user_all_data[name] = x1.get_group(user[0])
	usernames[name] = user[0]




# CREATE DATA STRUCTURE FOR TRAIN TEST AND ANAMOLY DATASET
train_data, test_data, anamoly_data = create_data_stucture(user_all_data, usernames)

# LOAD TEST TRAIN DATA

metric = 'Z_SCORE'
equal_error_rate_list = []
zero_false_alarm_list = []

for i,user in enumerate(usernames):
	# user = 'user4'
	train = train_data[user]
	test = test_data[user]
	anamoly = pd.DataFrame()

	# CREATE A LOOP TO CREATE THE ANAMOLY DATASET
	anamoly = create_anamoly_data_set_for_a_user(anamoly_data, user)

	# CREATE MEAN VECTOR
	mean = train.mean(axis = 0) 

	# CREATE MEAN VECTOR
	std = train.std(axis = 0) 

	# CALCULATE COVARIENCE MATRIX FROM TRAINING DATA
	cov_mat = train.T@train

	# MEAN ABSOLUTE DEVIATION OF EACH FEATURE
	abs_deviation = abs(train-mean)
	mean_abs_dev = np.sum(abs_deviation, axis=0)/train.shape[0]


	# CALCULATE z_score USER SCORE
	user_score, user_max, user_min = calculate_zscore_dist(test, mean, std)
	print(user+" test score - " + str(user_score.shape[0]) + " elemnts" )

	# CALCULATE z_score IMPOSTER SCORE
	anamoly_score, anamoly_max, anamoly_min = calculate_zscore_dist(anamoly, mean, std)
	print(user+" impo score - " + str(anamoly_score.shape[0]) + " elemnts" )

# -------------------------------------------------------------------------------
	
	# # CALCULATE MANHATTAN SCALED USER SCORE
	# user_score, user_max, user_min = calculate_manhattan_scaled_dist(test, mean, mean_abs_dev)
	# print(user+" test score - " + str(user_score.shape[0]) + " elemnts" )

	# # CALCULATE MANHATTAN SCALED IMPOSTER SCORE
	# anamoly_score, anamoly_max, anamoly_min = calculate_manhattan_scaled_dist(anamoly, mean, mean_abs_dev)
	# print(user+" impo score - " + str(anamoly_score.shape[0]) + " elemnts" )

	# # CALCULATE MANHATTAN USER SCORE
	# user_score, user_max, user_min = calculate_manhattan_dist(test, mean)
	# print(user+" test score - " + str(user_score.shape[0]) + " elemnts" )

	# # CALCULATE MANHATTAN IMPOSTER SCORE
	# anamoly_score, anamoly_max, anamoly_min = calculate_manhattan_dist(anamoly, mean)
	# print(user+" impo score - " + str(anamoly_score.shape[0]) + " elemnts" )

	# # CALCULATE MAHALANOBIS NORMED USER SCORE
	# user_score, user_max, user_min = calculate_mahalanobis_normed_dist(test, mean, cov_mat)
	# print(user+" test score - " + str(user_score.shape[0]) + " elemnts" )

	# # CALCULATE MAHALANOBIS IMPOSTER SCORE
	# anamoly_score, anamoly_max, anamoly_min = calculate_mahalanobis_normed_dist(anamoly, mean, cov_mat)
	# print(user+" impo score - " + str(anamoly_score.shape[0]) + " elemnts" )

	# # CALCULATE MAHALANOBIS USER SCORE
	# user_score, user_max, user_min = calculate_mahalanobis_dist(test, mean, cov_mat)
	# print(user+" test score - " + str(user_score.shape[0]) + " elemnts" )

	# # CALCULATE MAHALANOBIS IMPOSTER SCORE
	# anamoly_score, anamoly_max, anamoly_min = calculate_mahalanobis_dist(anamoly, mean, cov_mat)
	# print(user+" impo score - " + str(anamoly_score.shape[0]) + " elemnts" )

	# # CALCULATE EUCLEDIAN USER SCORE
	# user_score, user_max, user_min = (calculate_eucledian_dist(test, mean))
 	# print(user+" test score - " + str(user_score.shape[0]) + " elemnts" )

	# # CALCULATE IMPOSTER SCORE
	# anamoly_score, anamoly_max, anamoly_min = calculate_eucledian_dist(anamoly, mean)
	# print(user+" impo score - " + str(anamoly_score.shape[0]) + " elemnts" )
# ----------------------------------------------------------------------------------
	# CREATE SEARCHSPACE FOR OPTIMAL THRESHOLD
	hit_rate_list, miss_rate_list, flase_alarm_rate_list, error, equal_error_index, zero_miss_rate_index = search_optimal_threshold(user_score, anamoly_score)       

	# STORING THE EQUAL ERROR RATE FOR EACH USER
	equal_error_rate_list.append(miss_rate_list[equal_error_index])
	zero_false_alarm_list.append(flase_alarm_rate_list[zero_miss_rate_index])

	# PLOT THE ROC CURVE
	plot_ROC(i, user, hit_rate_list, flase_alarm_rate_list, equal_error_index, zero_miss_rate_index, metric)

equal_error_rate_list = np.stack(equal_error_rate_list)
zero_false_alarm_list = np.stack(zero_false_alarm_list)

error_rate_avg = np.mean(equal_error_rate_list)
error_rate_std = np.std(equal_error_rate_list) 

zero_miss_rate_avg = np.mean(zero_false_alarm_list)
zero_miss_rate_std = np.std(zero_false_alarm_list)

print("Equal error rate average = "+ str(error_rate_avg))
print("Equal error rate STD = "+ str(error_rate_std))

print("Zero miss false alarm rate average = "+ str(zero_miss_rate_avg))
print("Zero miss false alarm rate STD = "+ str(zero_miss_rate_std))