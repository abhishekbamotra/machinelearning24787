import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import shutil
np.random.seed(0)

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels





# CREATE DATA STRUCTURE FOR TRAIN TEST AND ANAMOLY DATASET

# THE DATA SET FOR EACH USER (HAVING 400 ROWS) ARE SHUFFLED FIRST AND THEN SPLIT

# DICTIONARY = "train_data" //  USER NAME AS KEY AND 200 DATA POINTS PANDAS DATAFRAME AS VALES
# EXAMPLE  = {user1 : PANDAS TRAIN DATAFRAME FOR USER i (200 ROWS)}

# DICTIONARY = "test_data" //  USER NAME AS KEY AND 200 DATA POINTS PANDAS DATAFRAME AS VALES
# EXAMPLE  = {user1 : PANDAS TEST DATAFRAME FOR USER i (200 ROWS)}

# DICTIONARY = "anamoly_data" //  USER NAME AS KEY AND 5 DATA POINTS PANDAS DATAFRAME AS VALES
# EXAMPLE  = {user1 : PANDAS ANAMOLY DATAFRAME FOR USER i (200 ROWS)}
# WE TAKE THE FIRST 5 ROWS OF THE TEST DATA SET TO CREATE THE ANAMOLY DATASET

def create_data_stucture(user_all_data, usernames):
	train_data = {}
	test_data = {}
	anamoly_data = {}

	# CREATE A LOOP TO GO OVER ALL USERS CREATE ABOVE DICTIONARIES
	for user in usernames.keys():

		# CREATE TRAIN, TEST, ANAMOLY_OTHER_USER DATA
		userdata = user_all_data[user]

		# DELETE "SUBJECT" COLUMN FROM THE DATA SET
		if 'subject' in userdata.columns:
		    del userdata['subject']

		# CREATE A TRAIN TEST SPLIT
		train, test =  train_test_split(userdata, train_size = 0.5, shuffle=True)

		# USING FIRST 5 ROWS FROM THE TEST SPLIT TO USE FOR ANAMOLY DATA GENERATION
		anamoly = test[:5]

		# ADDING THE TEST TRAIN AND ANAMOLY DATASET TO THE DICTIONARIES
		train_data[user] = train
		test_data[user] = test
		anamoly_data[user] = anamoly

	return train_data, test_data, anamoly_data






# CREATE DATA STRUCTURE COBINING DATA FROM ALL USERS FOR NEURAL NETWORK TRAINING

# FIRST THE DATA FRAMES OF ALL THE USERS (HAVING 400 ENTRIES EACH) ARE STCKED ONE AFTER THE OTHER
# SECOND A "Y" VECTOR IS CREATED WITH ONLY THE USER NUMBER (eg, "1", "2"...) FOR LABELS

# THE FUNCTION RETURNS 
# X - ALL THE X DATA (PANDAS DATA FRAME)
# Y - ALL THE LABEL DATA (PANDAS DATA FRAME)

def create_data_stucture_neural(user_all_data, usernames):

	# "user_all_data" = dictionary
	# "usernames" = dictionary

	x = pd.DataFrame()
	y = []

	# CREATE A LOOP TO GO OVER ALL USERS 
	for i,user in enumerate(usernames.keys()):

		userdata = user_all_data[user]
		# print("user", user, i)

		 # DELETE "SUBJECT" COLUMN FROM THE DATA SET
		if 'subject' in userdata.columns:
		    del userdata['subject']

		x = [x,userdata]
		x = pd.concat(x)

		# print(userdata.shape[0])
		y_tmp = np.ones((userdata.shape[0]))*i
		y.append(y_tmp)

	x = x.to_numpy()
	y = np.hstack(y)
	# print(y.shape, x.shape)
	# print(y)
	# print(x.shape)

	return x,y


def create_data_stucture_neural_negetive(user_all_data, usernames, neg_class):

	# "user_all_data" = dictionary
	# "usernames" = dictionary

	total_class = len(user_all_data.keys())
	neg_class_index = np.random.choice(total_class,neg_class,replace=False)

	# print(neg_class_index)

	x = pd.DataFrame()
	y = []
	x_anamoly = pd.DataFrame()
	y_anamoly = []
	new_user_index = 0

	# CREATE A LOOP TO GO OVER ALL USERS 
	for i,user in enumerate(usernames.keys()):

		userdata = user_all_data[user]
		# print("user", user, i)


		 # DELETE "SUBJECT" COLUMN FROM THE DATA SET
		if 'subject' in userdata.columns:
		    del userdata['subject']

		if np.isin(i,neg_class_index):
			x_anamoly = [x_anamoly,userdata]
			x_anamoly = pd.concat(x_anamoly)
		else:
			x = [x,userdata]
			x = pd.concat(x)
			y_tmp = np.ones((userdata.shape[0]))*new_user_index
			y.append(y_tmp)
			new_user_index = new_user_index+1


	x = x.to_numpy()

	x_anamoly = x_anamoly.to_numpy()
	np.random.shuffle(x_anamoly)
	
	x_anamoly = x_anamoly[0:userdata.shape[0],:]
	y_anamoly = np.ones((userdata.shape[0]))*new_user_index

	x = np.vstack((x,x_anamoly))

	y.append(y_anamoly)
	y = np.hstack(y)

	return x,y



def create_anamoly_data_set_for_a_user(anamoly_data, user):
	anamoly = pd.DataFrame()
	for tmp_user in anamoly_data:
		if tmp_user!=user:
			anamoly = [anamoly, anamoly_data[tmp_user]]
			anamoly = pd.concat(anamoly)
	return anamoly



def plot_ROC(i, user, hit_rate_list, flase_alarm_rate_list, equal_error_index, zero_miss_rate_index, metric):
	

	folder_path_for_plots = ("plots/"+metric)

	# IF ANY PREVIOUS IMAGES ALREADY EXISTS REMOVE THEM
	# DO THIS ONLY FOR THE FIRST LOOP WHEN i=0
	# if i==0:
	# 	shutil.rmtree(folder_path_for_plots)
	# # print(i)
	os.makedirs(folder_path_for_plots, exist_ok=True)

	# GENERATE ROC PLOT
	plt.plot(flase_alarm_rate_list, hit_rate_list, label = 'ROC curve')
	plt.scatter(flase_alarm_rate_list[equal_error_index], hit_rate_list[equal_error_index],
		        marker='o', s=20, linewidths=5,
		        color='r', zorder=10, label='Equal Error Rate')
	plt.scatter(flase_alarm_rate_list[zero_miss_rate_index], hit_rate_list[zero_miss_rate_index],
		        marker='o', s=20, linewidths=5,
		        color='g', zorder=10, label='Zero Miss Rate')
	title = str('ROC CURVE _ METRIC_'+ metric + ' _ ' + user)
	title = str('ROC CURVE _ METRIC_'+ metric + ' _ ' + user)
	plt.title(title)
	plt.xlabel('False error rate')
	plt.ylabel('Hit rate')
	plt.grid()
	plt.legend(loc = 'lower right')

	image_name = (str("{:02d}".format(i+1)) +'_'+ title+".jpg")
	image_path = os.path.join(folder_path_for_plots, image_name)
	plt.savefig(image_path, dpi=300, bbox_inches='tight')
	plt.close()
	




def search_optimal_threshold(user_score, anamoly_score):
	max_score = max(user_score.max(), anamoly_score.max())
	min_score = min(user_score.min(), anamoly_score.min())

	# CREATE SEARCHSPACE FOR OPTIMAL THRESHOLD
	threshold = np.linspace(min_score, max_score, 1000)

	# CREATE BLANK LISTS FOR HIT RATE, MISS RATE, FALSE ALARM RATE
	hit_rate_list=[]
	miss_rate_list = []
	flase_alarm_rate_list = []
	error = 1000
	equal_error_index = 0
	zero_miss_rate_index = 0
	zero_miss_flag = 0


	# LOOP OVER ALL THE THREASHOLD VALUES TO GET HIT AND MISS RATE
	for i,tmp_thresh in enumerate(threshold):
		# tmp_thresh = 0.6

		# MISS RATE =  FREQ WITH WHICH IMPOSTERS ARE NOT DETECTED
		# CALCULATE MISS RATE
		imposter_user = anamoly_score.shape[0]
		imposters_deceted = sum((anamoly_score>tmp_thresh)*1)
		hit_rate = imposters_deceted/imposter_user
		miss_rate = 1-hit_rate
		# print(miss_rate)
		hit_rate_list.append(hit_rate)
		miss_rate_list.append(miss_rate)
		# print(imposter_user)
		# print("miss_rate", miss_rate)

		# FALSE ALARM RATE - FREQ WITH WHICH GENUINE USERS ARE MISTAKENLY DETECTED
		# CALCULATE FALSE ALARM RATE (GENUINE USERS MISTAKENLY DETECTED AS ANAMOLY)
		genuine_users = user_score.shape[0]
		genuine_user_false_detection = sum((user_score>tmp_thresh)*1)
		false_alarm_rate = genuine_user_false_detection/genuine_users
		flase_alarm_rate_list.append(false_alarm_rate)
		# print("miss_rate", miss_rate, "false_alarm_rate", false_alarm_rate)

		# CALCULTE RATIO OF FALSE ALARM RATE / MISS RATE
		# CHECK IF IT IS CLOSEST TO ONE

		# FOR NUMERICAL STABILITY AVOID DIVISION BY ZERO
		if (miss_rate==0.0 or false_alarm_rate==0.0):
		    ratio = 100
		else:
		    ratio = miss_rate/false_alarm_rate

		# CALCULATING HOW CLOSE THE RATIO IS TO ONE
		# FINDING THE EQUAL ERROR RATE 
		tmp_error = abs(1 - ratio)
		if tmp_error<error:
		    equal_error_index = i
		    error = tmp_error

		# CALCULATE THE INDEX FOR ZERO MISS RATE
		# print(i, miss_rate) 
		if zero_miss_flag == 0:
			if miss_rate >0:
				zero_miss_rate_index = i
				zero_miss_flag =1






	# print(hit_rate_list)
	# print("zero_miss_rate_index", zero_miss_rate_index)
	return hit_rate_list, miss_rate_list, flase_alarm_rate_list, error, equal_error_index, zero_miss_rate_index


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100
        cm = cm.astype('int')
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    # plt.figure(figsize=(15,4))
    fig, ax = plt.subplots(figsize=(16,16))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    fmt = 'd' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    np.set_printoptions(precision=2)

    plt.savefig("./conv_1d_confusionmatrix.jpg", dpi=300, bbox_inches='tight')
    plt.show()
    return ax,cm


	




def generate_confusion_matrix(y_true,y_pred,class_names):
    cm =confusion_matrix(y_true, y_pred)


    import string
    fig = plt.figure(4, figsize=(10,10))
    plt.imshow(cm,interpolation='nearest')

    plt.grid(True)
    plt.xticks(class_names)
    plt.yticks(class_names)
    plt.title("Confusion_matrix")
    # if plot_flag:
    plt.savefig("./results/confusion_matrix.jpg", dpi=300, bbox_inches='tight')
    plt.show()
    return cm