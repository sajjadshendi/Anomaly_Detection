#Import necessary libraries for the elliptic envelope algorithm
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import numpy as np

#Define a class for elliptic envelope itself
class elliptic_envelope:
    def __init__(self, raw_df, df, anomaly_threshold, flag = False):
        self.criterion_column = i
        self.anomaly_condition = None
        #Define refined dataframes
        self.df = df
        self.raw_df = raw_df
        # define parameters of elliptic_envelope
        self.anomaly_threshold = anomaly_threshold
        self.flag = flag
        self.support_fraction = 1
        ###
        # define and split train, validation and test data
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

    #Define a function that runs the elliptic envelope on the whole dataset and predicts anomalies. On that, then it returns anomaly indexes, raw data of anomaly rows and whole predictions
    def detector(self):
        # Define raw data of the whole refining dataframe
        df_array = self.df.values
        indexes = np.array([])
        anomalies = np.array([])
        # Define and run the elliptic envelope
        ee = EllipticEnvelope(contamination = self.anomaly_threshold, support_fraction = self.support_fraction)
        ee.fit(self.df)
        y_pred = ee.predict(self.df)
        # Obtain indexes and information about anomalies
        for i in range(len(y_pred)):
            if(y_pred[i] == -1):
                indexes = np.append(indexes, i)
                anomalies = np.append(anomalies, df_array[i])
        return indexes, anomalies, y_pred

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

    # A function that fits a model on train data then returns it
    def train(self, df1, df2):
        length = len(self.raw_data2)
        anomals = self.label_maker(self.raw_data2)
        anomaly_threshold = len(anomals) / length
        model = EllipticEnvelope(contamination=anomaly_threshold, support_fraction = self.support_fraction)
        model.fit(df1)
        best_model = model
        return best_model

    # A function that examines tuned model on the test dataframe for getting anomaly indexes in the test dataframe
    def test(self, df3, best_model):
        indexes = np.array([])
        test_anomalies = []
        df3_array = df3.values
        model = best_model
        predictions = model.predict(df3)
        for i in range(len(predictions)):
            if(predictions[i] == -1):
                indexes = np.append(indexes, i)
                test_anomalies.append(df3_array[i])
        test_indexes = indexes + df3.index[0]
        return test_indexes, test_anomalies

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
    def detection(self, ):
        best_model = self.train(self.data1, self.data2)
        test_indexes, test_anomalies = self.test(self.data3, best_model)
        actual_anomals = self.label_maker(self.raw_data3)
        precision, recall, f1_score, false_positive_rate, false_negative_rate = self.measurements(test_indexes,
                                                                                                  actual_anomals)
        if (self.flag):
            indexes, anomalies, preds = self.detector()
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
