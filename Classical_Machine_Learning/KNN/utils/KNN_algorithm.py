#Import necessary libraries for the KNN algorithm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

#Define a class for KNN itself
class knn:
    def __init__(self, raw_data, data, flag = False):
        self.criterion_column = i
        self.anomaly_condition = anomaly_condition
        # Define refined dataframes
        self.df = data
        self.raw_df = raw_data
        # Define hyperparameters of KNN
        self.n_neighbors = 10
        self.percentile_threshold = 99
        # Define and split train, validation and test data
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.total_rows = self.df.shape[0]
        ###
        self.train_data = self.df[:int(self.total_rows * self.train_ratio)]
        self.val_data = self.df[int(self.total_rows * self.train_ratio): int(
            self.total_rows * (self.train_ratio + self.val_ratio))]
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
        ###
        self.flag = flag

    # Define a function that runs KNN on the whole dataset and predicts anomalies. On that, then it returns anomaly indexes, raw data of anomaly rows and mean distance for every data. Also it returns predictions
    def detector(self, df):
        # Define raw data of the whole refined dataframe
        values = df.values
        # Define and run KNN
        knn_op = NearestNeighbors(n_neighbors=self.n_neighbors)
        knn_op.fit(values)
        dists, inds = knn_op.kneighbors(values)
        dists = dists.mean(axis=1)
        anomalies = []
        indexes = np.array([])
        distance_threshold = np.percentile(dists, self.percentile_threshold)
        # Obtain indexes and information about anomalies
        for i in range(len(dists)):
            if (dists[i] > distance_threshold):
                indexes = np.append(indexes, i)
                anomalies.append(values[i])
        ###
        preds = np.ones(len(df))
        for j in indexes:
            preds[int(j)] = -1
        ###
        return indexes, anomalies, dists, preds

    # A function that detects truly anomaly indexes
    def label_maker(self, df):
        columns = df.columns
        anomals = np.array(df.loc[self.anomaly_condition].index)
        return anomals

    # A function that gets true and predicted anomaly indexes then returns some evaluations of the model
    def measurements(self, indexes, anomals):
        if(len(indexes) == 0):
            return 0, 0, 0, 1, 1
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

    # A function that fits a model on train data then validates it on validation data to obtain the proper n_neighbors. Then it returns the best model that runs the best n_neighbors and best threshold
    def train(self, df1, df2):
        df1_values = df1.values
        df2_values = df2.values
        anomals = self.label_maker(self.raw_data2)
        # calculate proper percentile threshold
        length = len(self.raw_data2)
        percentile_threshold = (1 - (len(anomals) / length)) * 100
        best_threshold = percentile_threshold
        ###
        best_f1_score = 0
        best_model = None
        tmp_f1_score = 0
        # validate different amounts for percentile_threshold
        for n_neighbors in np.arange(5, 30, 5):
            model = NearestNeighbors(n_neighbors=n_neighbors)
            model.fit(df1_values)
            dists, inds = model.kneighbors(df2_values)
            dists = dists.mean(axis=1)
            tmp_indexes = np.array([])
            distance_threshold = np.percentile(dists, percentile_threshold)
            for i in range(len(dists)):
                if (dists[i] > distance_threshold):
                    tmp_indexes = np.append(tmp_indexes, i)
            tmp_indexes = tmp_indexes + df2.index[0]
            # Validate model with the specified threshold for getting the best f1 score
            precision, recall, f1_score, false_positive_rate, false_negative_rate = self.measurements(tmp_indexes,
                                                                                                          anomals)
            if (best_f1_score < f1_score):
                best_model = model
                best_f1_score = f1_score
                best_threshold = percentile_threshold

            if (tmp_f1_score < f1_score):
                tmp_f1_score = f1_score
            else:
                break

        return best_model, best_threshold
        
    # A function that examines tuned model on test dataframe for getting anomaly indexes in the test dataframe
    def test(self, df3, best_model, best_threshold):
        indexes = np.array([])
        test_anomalies = []
        df3_values = df3.values
        model = best_model
        dists, inds = model.kneighbors(df3_values)
        dists = dists.mean(axis = 1)
        distance_threshold = np.percentile(dists, best_threshold)
        for i in range(len(dists)):
            if(dists[i] > distance_threshold):
                indexes = np.append(indexes, i)
                test_anomalies.append(df3_values[i])
        test_indexes = indexes + df3.index[0]
        return test_indexes, test_anomalies

    # A function for showing anomalies of detector function based on indexes and availability to show that anomalies have less availability
    def show(self, df, preds):
        normal_count = len(np.where(preds == 1)[0])
        anomaly_count = len(np.where(preds == -1)[0])
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        axes.scatter(df[preds == -1].iloc[0:len(preds):100, :].index,
                        df[preds == -1].iloc[0:len(preds):100, :].iloc[:, self.criterion_column], color='red',
                        label='Anomaly: ' + str(anomaly_count))
        axes.scatter(df[preds == 1].iloc[0:len(preds):10000, :].index,
                        df[preds == 1].iloc[0:len(preds):10000, :].iloc[:, self.criterion_column],
                        color='green', label='Normal: ' + str(normal_count))
        axes.legend(loc='best')
        plt.tight_layout()
        plt.show()

    # A function that first runs algorithm on train, validation and test data, then runs it on the whole dataset to show some results
    def detection(self,):
        # training the model
        best_model, best_threshold = self.train(self.data1, self.data2)
        test_indexes, test_anomalies = self.test(self.data3, best_model, best_threshold)
        actual_anomals = self.label_maker(self.raw_data3)
        precision, recall, f1_score, false_positive_rate, false_negative_rate = self.measurements(test_indexes, actual_anomals)

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


