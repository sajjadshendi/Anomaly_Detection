#Import necessary libraries for the Linear Regression algorithm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import copy

#Define a class for Linear Regression itself
class lin_reg_alg:
    def __init__(self, raw_df, df, minimum_threshold, percentile_threshold, flag = False):
        self.criterion_column = i
        self.enterance_part = enterance_part
        self.anomaly_condition = anomaly_condition
        self.labels = labels
        self.flag = flag
        # Define refined dataframes
        self.raw_df = raw_df
        self.df = df
        #Defin eparameters of Linear Regression
        self.minimum_threshold = minimum_threshold
        self.percentile_threshold = percentile_threshold
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

    # Define a function that runs Linear Regression on the whole dataset and predicts anomalies. On that, then it returns anomaly indexes, raw data of anomaly rows and distance from the hyperplane for every data. Also it returns predictions
    def detector(self, df):
        #Define raw data of the whole refined dataframe
        df_array = df.values
        #Define and run Linear Regression
        fitting_data = df_array[self.enterance_part]
        reg = LinearRegression()
        reg = reg.fit(fitting_data, df_array[self.labels])
        lin_reg_predictions = reg.predict(fitting_data)
        distances = (lin_reg_predictions - df_array[self.labels]) ** 2
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

    # A function that detects truly anomaly indexes
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

    #A function that gets a fitted model that was fitted on train data then validates it on validation data to obtains proper percentile_threshold and returns the best threshold for the testing part
    def validate(self, linear_regression, dataset, df):
        lin_reg_predictions = linear_regression.predict(df[self.enterance_part])
        val_distances = (lin_reg_predictions - df[self.labels]) ** 2
        best_threshold = 0
        best_f1_score = 0
        actual_anomals = self.label_maker(self.raw_data2)
        final_actual_anomals = actual_anomals - dataset.index[0]
        #calculate proper percentile threshold
        length = len(self.raw_data2)
        percentile_threshold = (1 - (len(actual_anomals) / length)) * 100
        val_threshold_distance = np.percentile(val_distances, percentile_threshold)
        val_anomalies = []
        val_indexes = np.array([])
        for i, val_distance in enumerate(val_distances):
            if val_distance > val_threshold_distance:
                val_anomalies.append(df[i])
                val_indexes = np.append(val_indexes, i)
        #Evaluate model with the specified threshold for getting the best f1 score
        val_precision, val_recall, val_f1_score, val_false_positive_rate, val_false_negative_rate = self.measurements(val_indexes, final_actual_anomals)
        if(best_f1_score < val_f1_score):
            best_threshold = percentile_threshold
            best_f1_score = val_f1_score
        #Return best threshold that gives the best f1 score to us
        return best_threshold
        
    #A function that examines tuned model on a test dataframe for getting anomaly indexes in that
    def test_valid(self, df1, df2, df3):
        df1_array = df1.values
        df2_array = df2.values
        df3_array = df3.values
        fitting_data = df1_array[self.enterance_part]
        lin_reg_model = LinearRegression().fit(fitting_data, df1_array[self.labels])
        #Run validation step for getting the best threshold
        best_threshold = self.validate(lin_reg_model, df2, df2_array)
        test_linreg_predictions = lin_reg_model.predict(df3_array[self.enterance_part])
        test_distances = (test_linreg_predictions - df3_array[self.labels]) ** 2
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

    #A function that first runs algorithm on train, validation and test data, then runs it on the whole dataset to show some results
    def detection(self,):
        test_indexes, test_anomalies, test_distances = self.test_valid(self.data1, self.data2, self.data3)
        actual_anomals = self.label_maker(self.raw_data3)
        final_actual_anomals = actual_anomals - self.data3.index[0]
        precision, recall, f1_score, false_positive_rate, false_negative_rate = self.measurements(test_indexes, final_actual_anomals)

        if(self.flag):
            indexes, anomalies, distances, preds = self.detector(self.df)
            self.show(self.raw_df, preds)
            self.show(indexes, anomalies, distances, 'line')
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
