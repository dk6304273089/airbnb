from extract_data import read_params,log
import pandas as pd
import re
import argparse
file = open("Training_logs/Training_log.txt", "a+")
c=["longitude","minimum_nights","price"]
def outlier(d,x):
    upper_limit = d["{}".format(x)].mean() + 3 * d["{}".format(x)].std()
    print(upper_limit)
    lower_limit = d["{}".format(x)].mean() - 3 * d["{}".format(x)].std()
    print(lower_limit)
    d = d[(d["{}".format(x)] > lower_limit) & (d["{}".format(x)] < upper_limit)]
    log(file, "Stage-3 => successfully Removed outliers for {}".format(x))
    return d
def outlier_cities(df):
    cities = list(df["city"].value_counts().index)
    for i in cities:
        df1 = df[df["city"] == "{}".format(i)]
        df = df[df["city"] != "{}".format(i)]
        upper_limit = df1.latitude.mean() + 3 * df1.latitude.std()
        lower_limit = df1.latitude.mean() - 3 * df1.latitude.std()
        log(file, "Stage-3 => successfully Removed outliers for {}".format(i))
        df1 = df1[(df1.latitude < upper_limit) & (df1.latitude > lower_limit)]
        df = df.append(df1)
    return df

def get_data(config_path):
    config = read_params(config_path)
    # print(config)
    data_path = config["Data"]["Processed"]
    df = pd.read_csv(data_path, sep=",")
    return df

class feature:
    def __init__(self):
        self.file=open("Training_logs/Training_log.txt","a+")
    def engineering(self, config_path):
        try:
            #removing outliers
            config = read_params(config_path)
            data=config["Data"]["Evaluated"]
            df = get_data(config_path)
            for i in c:
                df = outlier(df, i)
            df=outlier_cities(df)
            df = df[df["price"] < 400]
            df.to_csv(data,index=False)
            log(self.file, "Stage-3 => successfully Done Feature Engineering and saved the file in {}".format(data))
        except Exception as e:
            log(self.file, "Stage-3 => Error: {}".format(str(e)))

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="config.yaml")
    parsed_args=args.parse_args()
    feature().engineering(config_path=parsed_args.config)



