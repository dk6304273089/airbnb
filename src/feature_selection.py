import argparse
from extract_data import read_params,log
from sklearn import preprocessing
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

def get_data(config_path):
    config = read_params(config_path)
    # print(config)
    data_path = config["Data"]["Evaluated"]
    df = pd.read_csv(data_path, sep=",")
    return df

class feature:
    def __init__(self):
        self.file=open("Training_logs/Training_log.txt","a+")
    def selection(self, config_path):
        try:
            #feature selection and label encoder steps
            config = read_params(config_path)
            data=config["Data"]["Final"]
            city_model=config["models"]["Labelencoder"]["city"]
            property_type_model=config["models"]["Labelencoder"]["property_type"]
            room_type_model=config["models"]["Labelencoder"]["room_type"]
            cancellation_policy_model=config["models"]["Labelencoder"]["cancellation_policy"]
            df = get_data(config_path)
            city = preprocessing.LabelEncoder()
            city.fit(df["city"])
            df["city"] = city.transform(df["city"])
            pickle.dump(city, open(city_model, 'wb'))
            log(self.file, "Stage-4 => successfully saved model for city in {}".format(city_model))
            property_type = preprocessing.LabelEncoder()
            property_type.fit(df["property_type"])
            df["property_type"] = property_type.transform(df["property_type"])
            pickle.dump(property_type, open(property_type_model, 'wb'))
            log(self.file, "Stage-4 => successfully saved model for property_type in {}".format(property_type_model))
            room_type = preprocessing.LabelEncoder()
            room_type.fit(df["room_type"])
            df["room_type"] = room_type.transform(df["room_type"])
            pickle.dump(room_type, open(room_type_model, 'wb'))
            log(self.file, "Stage-4 => successfully saved model for room_type in {}".format(room_type_model))
            cancellation_policy = preprocessing.LabelEncoder()
            cancellation_policy.fit(df["cancellation_policy"])
            df["cancellation_policy"] = cancellation_policy.transform(df["cancellation_policy"])
            pickle.dump(cancellation_policy, open(cancellation_policy_model, 'wb'))
            log(self.file, "Stage-4 => successfully saved model for cancellation_policy in {}".format(cancellation_policy_model))
            df["host_identity_verified"] = df["host_identity_verified"].apply(lambda x: 1 if x == 't' else 0)
            df["host_is_superhost"] = df["host_is_superhost"].apply(lambda x: 1 if x == 't' else 0)
            #handling Missing Values
            imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            imputer = imputer.fit(df)
            X = imputer.transform(df)
            df = pd.DataFrame(X, columns=df.columns)
            log(self.file,
                "Stage-4 => successfully Handled missing values for the total data")
            #Adding new columns to get more information related to data
            df["bedroom_per_person"] = df["bedrooms"] / df["accommodates"]
            df["bathroom_per_person"] = df["bathrooms"] / df["accommodates"]
            log(self.file,"Stage-4 => successfully Added new columns ")
            #independent columns
            X = df.drop(['price'], axis=1)
            #target column
            y = df['price']
            #performing feature selection to get only top 20 important columns that are correlated with target variable
            mc = MinMaxScaler()
            x = mc.fit_transform(X)
            x = pd.DataFrame(x, columns=X.columns)
            ordered_rank_features = SelectKBest(score_func=chi2, k=20)
            ordered_feature = ordered_rank_features.fit(x, y)
            dfscores = pd.DataFrame(ordered_feature.scores_, columns=["Score"])
            dfcolumns = pd.DataFrame(X.columns)
            features_rank = pd.concat([dfcolumns, dfscores], axis=1)
            features_rank.columns = ['Features', 'Score']
            h = pd.DataFrame(features_rank.nlargest(20, 'Score'))
            log(self.file, "Stage-4 => successfully performed feature selection")
            d = list(h["Features"])
            df = df[d]
            df["price"]=y
            df.to_csv(data,index=False)
            log(self.file, "Stage-4 => successfully saved the final data in {}".format(data))
        except Exception as e:
            log(self.file, "Stage-3 => Error: {}".format(str(e)))

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="config.yaml")
    parsed_args=args.parse_args()
    feature().selection(config_path=parsed_args.config)




