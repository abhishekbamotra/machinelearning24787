import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from utils import *
np.random.seed(0)



# FUNCTION FOR EUCLEDIAN DISTANCE CALCULATION
def calculate_eucledian_dist(test, mean):
	distance = (test - mean).pow(2)
	score = (distance.sum(axis=1)).pow(0.5)
	return score, score.max(), score.min()



# FUNCTION FOR MAHALANOBIS DISTANCE CALCULATION
def calculate_mahalanobis_dist(test, mean, cov_mat):
	diff = test-mean
	cov_inv = np.linalg.inv(cov_mat)
	difft = diff.T
	score1 = np.dot(diff,cov_inv)
	score2 = np.dot(score1, difft)
	score = np.diag(score2)
	return score, score.max(), score.min()



# FUNCTION FOR MAHALANOBIS NORMED DISTANCE CALCULATION
def calculate_mahalanobis_normed_dist(test, mean, cov_mat):
	mean_norm = np.linalg.norm(mean)
	test_norm = np.linalg.norm(test, axis=1)
	norm_product = np.multiply(mean_norm, test_norm)

	# print("norm_product",norm_product.shape)
	diff = test-mean
	cov_inv = np.linalg.inv(cov_mat)
	difft = diff.T
	score1 = np.dot(diff,cov_inv)
	score2 = np.dot(score1, difft)
	score3 = np.diag(score2)
	# print("score3", score3.shape)
	score = np.divide(score3, norm_product)
	# print("score", score.shape)
	return score, score.max(), score.min()



# FUNCTION FOR MANHATTAN DISTANCE CALCULATION
def calculate_manhattan_dist(test, mean):
	distance = abs(test-mean)
	score = (distance.sum(axis=1))
	return score, score.max(), score.min()




# FUNCTION FOR MANHATTAN SCALED DISTANCE CALCULATION
def calculate_manhattan_scaled_dist(test, mean, mean_abs_dev):
	distance = abs(test-mean)
	scaled_dist = np.divide(distance,mean_abs_dev)
	score = (scaled_dist.sum(axis=1))
	return score, score.max(), score.min()


# FUNCTION FOR ZSCORE  DISTANCE CALCULATION
def calculate_zscore_dist(test, mean, std):
	distance = abs(test-mean)
	scaled_dist = np.divide(distance,std)
	score = (scaled_dist.sum(axis=1))
	return score, score.max(), score.min()