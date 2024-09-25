#Import necessary libraries for the kmeans algorithm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
import pandas as pd
import numpy as np
import copy

#Define a class for kmeans itself
class kmeans_algorithm:
    def __init__(self, raw_df, df, maximum_cluster, minimum_threshold, percentile_threshold, flag = False):
        self.criterion_column = i
        self.anomaly_condition = None
        self.flag = flag
        #Define refined dataframes
        self.df = df
        self.raw_df = raw_df
        #Define parameters of kmeans
        self.maximum_cluster = maximum_cluster
        self.minimum_threshold = minimum_threshold
        self.percentile_threshold = percentile_threshold
        #Define and split train, validation and test data
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.total_rows = self.df.shape[0]
        ###
        self.train_data = self.df[:int(self.total_rows * self.train_ratio)]
        self.val_data = self.df[int(self.total_rows * self.train_ratio): int(self.total_rows * (self.train_ratio + self.val_ratio))]
        self.test_data = self.df[int(self.total_rows * (self.train_ratio + self.val_ratio)):]
        self.data1 = self.train_data.copy(deep=True)
        self.data2 = self.val_data.copy(deep=True)
        self.data3 = self.test_data.copy(deep=True)
        ###
        self.raw_train_data = self.raw_df[:int(self.total_rows * self.train_ratio)]
        self.raw_val_data = self.raw_df[int(self.total_rows * self.train_ratio): int(
            self.total_rows * (self.train_ratio + self.val_ratio))]
        self.raw_test_data = self.raw_df[int(self.total_rows * (self.train_ratio + self.val_ratio)):]
        self.raw_data1 = self.raw_train_data.copy(deep=True)
        self.raw_data2 = self.raw_val_data.copy(deep=True)
        self.raw_data3 = self.raw_test_data.copy(deep=True)


    #Define a function that runs kmenas on the whole dataset and predicts anomalies. On that, then it returns anomaly indexes, raw data of anomaly rows and distances of points from their centers. Also it returns predictions
    def detector(self, df):
        #Define raw data of the whole refined dataframe
        df_array = df.values
        #Obtain proper number of clusters
        inertias = []
        for i in range(1, 5):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(df_array)
            inertias.append(kmeans.inertia_)
        tmp = range(1, len(inertias) + 1)
        kn = KneeLocator(tmp, inertias, curve='convex', direction='decreasing')
        num_cluster = kn.knee
        if(num_cluster == None):
            num_cluster = kn.knee = 2
        #Define and run kmeans
        kmeans = KMeans(n_clusters=num_cluster)
        kmeans.fit(df_array)
        kmeans_labels = kmeans.predict(df_array)
        cluster_centers = kmeans.cluster_centers_
        distances = [np.linalg.norm(df_array[i] - cluster_centers[kmeans_labels[i]]) for i in range(len(kmeans_labels))]
        threshold_distance = np.percentile(distances, self.percentile_threshold)
        anomalies = []
        indexes = np.array([])
        #Obtain indexes and information about anomalies
        for i, distance in enumerate(distances):
            if distance > threshold_distance:
                anomalies.append(df_array[i])
                indexes = np.append(indexes, i)
        ###
        preds = np.ones(len(df))
        for j in indexes:
            preds[int(j)] = -1
        ###
        return indexes, anomalies, distances, preds

    #A function that obtains proper number of clusters
    def knee_locator(self, df, maximum_cluster):
        inertias = []
        for i in range(1, maximum_cluster):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(df)
            inertias.append(kmeans.inertia_)
        tmp = range(1, len(inertias)+1)
        kn = KneeLocator(tmp, inertias, curve='convex', direction='decreasing')
        return kn.knee

    #A function that runs kmenas and returns a fitted model
    def fitting(self, df, num_cluster):
        kmeans = KMeans(n_clusters = num_cluster)
        kmeans.fit(df)
        return kmeans

    #A function that detects truly anomaly indexes
    def label_maker(self, df):
        columns = df.columns
        anomals = np.array(df.loc[self.anomaly_condition].index)
        return anomals

    #A function that gets true and predicted anomaly indexes then returns some evaluations of the model
    def measurements(self, indexes, anomals):
        true_positive = len(np.intersect1d(indexes, anomals))
        false_negative = len([value for value in anomals if not(value in indexes)])
        false_positive = len([value for value in indexes if not(value in anomals)])
        #print(true_positive, false_positive, false_negative)
        precision = true_positive / (true_positive + false_negative)
        recall = true_positive / (true_positive + false_positive)
        f1_score = 2 * (precision * recall) / (precision + recall)
        false_positive_rate = false_positive / len(indexes)
        false_negative_rate = false_negative / len(anomals)
        return precision, recall, f1_score, false_positive_rate, false_negative_rate

    #A function that examines tuned model on a test dataframe for getting anomaly indexes in that
    def test_valid(self, df1, df2, df3):
        df1_array = df1.values
        df2_array = df2.values
        df3_array = df3.values
        num_cluster = self.knee_locator(df1_array, self.maximum_cluster)
        kmeans = self.fitting(df1_array, num_cluster)
        cluster_centers = kmeans.cluster_centers_
        #calculate best_threshold
        length = len(self.raw_data2)
        anomals = self.label_maker(self.raw_data2)
        percentile_threshold = (1 - (len(anomals) / length)) * 100
        best_threshold = percentile_threshold
        ###
        test_kmeans_labels = kmeans.predict(df3_array)
        test_distances = [np.linalg.norm(df3_array[i] - cluster_centers[test_kmeans_labels[i]]) for i in range(len(test_kmeans_labels))]
        test_threshold_distance = np.percentile(test_distances, best_threshold)
        test_anomalies = []
        test_indexes = np.array([])
        for i, test_distance in enumerate(test_distances):
            if test_distance > test_threshold_distance:
                test_anomalies.append(df3_array[i])
                test_indexes = np.append(test_indexes, i)
        return test_indexes, test_anomalies, test_distances

    # A function for showing anomalies of detector function based on indexes and availability to show that anomalies have less availability
    def show(self, df, preds):
        normal_count = len(np.where(preds ==1)[0])
        anomaly_count = len(np.where(preds == -1)[0])
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        axes.scatter(df[preds == -1].iloc[0:len(preds):100, :].index,
                        df[preds == -1].iloc[0:len(preds):100, :].iloc[:, self.criterion_column], color='red', label='Anomaly: ' + str(anomaly_count))
        axes.scatter(df[preds == 1].iloc[0:len(preds):10000, :].index,
                        df[preds == 1].iloc[0:len(preds):10000, :].iloc[:, self.criterion_column],
                        color='green', label='Normal: ' + str(normal_count))
        axes.legend(loc='best')
        plt.tight_layout()
        plt.show()

    #A function that first runs algorithm on train, validation and test data, then runs it on the whole dataset to show some results
    def detection(self,):
        test_indexes, test_anomalies, test_distances = self.test_valid(self.data1, self.data2, self.data3)
        actual_anomals = self.label_maker(self.raw_data3)
        final_actual_anomals = actual_anomals - self.data3.index[0]
        precision, recall, f1_score, false_positive_rate, false_negative_rate = self.measurements(test_indexes, final_actual_anomals)

        if(self.flag):
            indexes, anomalies, distances, preds = self.detector(self.df)
            self.show(self.raw_df, preds)
            anomals = self.label_maker(self.raw_df)
            total_precision, total_recall, total_f1_score, total_false_positive_rate, total_false_negative_rate = self.measurements(
                indexes, anomals)
            print("without seperation we have:")
            print("precision is: ", total_precision)
            print("recall is: ", total_recall)
            print("f1 score is: ", total_f1_score)
            print("false positive rate is: ", total_false_positive_rate)
            print("false negative rate is: ", total_false_negative_rate)
            print("with seperation we have:")
            print("test precision is: ", precision)
            print("test recall is: ", recall)
            print("test f1 score is: ", f1_score)
            print("test false positive rate is: ", false_positive_rate)
            print("test false negative rate is: ", false_negative_rate)
            return f1_score
        else:
            return f1_score
