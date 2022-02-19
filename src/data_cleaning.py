from extract_data import read_params,log
import pandas as pd
import re
import argparse
import re
price_list = ["price", "cleaning_fee", "security_deposit"]
include_Cols=["city","longitude","latitude","review_scores_rating","number_of_reviews","minimum_nights","accommodates","bathrooms","bedrooms","beds","security_deposit","cleaning_fee","property_type","room_type","availability_365","host_identity_verified","host_is_superhost","cancellation_policy","price","zipcode","state"]
def get_data(config_path):
    config = read_params(config_path)
    # print(config)
    data_path = config["Data"]["Raw"]
    df = pd.read_csv(data_path, sep=",")
    return df
class data:
    def __init__(self):
        self.file=open("Training_logs/Training_log.txt","a+")
    def clean_data(self, config_path):
        try:
            #removing unwanted columns from Data
            config = read_params(config_path)
            data=config["Data"]["Processed"]
            df = get_data(config_path)
            df=df[include_Cols]
            log(self.file, "Stage-2 => successfully Removed unwanted columns")
            list_of_20 = list(df["city"].value_counts().head(15).index)
            df = df[df["city"].isin(list_of_20)].reset_index(drop=True)
            df = df[(df["zipcode"] != "NSW 2022") & (df["zipcode"] != "NSW 2025")]
            df["zipcode"] = df["zipcode"].astype(float)
            list_of_20 = list(df["zipcode"].value_counts().head(12).index)
            df = df[df["zipcode"].isin(list_of_20)].reset_index(drop=True)
            df["zipcode"] = df["zipcode"].astype(int)
            for col in price_list:
                df[col] = df[col].fillna("0")
                df[col] = df[col].apply(lambda x: float(re.compile('[^0-9eE.]').sub('', x)) if len(x) > 0 else 0)
            df["city"] = df["city"].replace(
                {"Bondi": "Bondi Beach", "North Bondi": "Bondi Beach", "Bondi Junction": "Bondi Beach"})
            df["state"] = df["state"].replace({"New South Wales": "NSW", "Nsw": "NSW", "Now": "NSW"})
            df = df[df["state"] != "Randwick"]
            df.drop(["state"], axis=1, inplace=True)
            c = df[(df["latitude"] < -33.800) & (df["latitude"] > -33.825) & (df["longitude"] < 151.20)].index
            df.drop(index=c, axis=0, inplace=True)
            df.to_csv(data, index=False)
            log(self.file, "Stage-2 => successfully Done Data cleaning and saved the file in {}".format(data))
        except Exception as e:
            log(self.file,"Stage-2 => Error: {}".format(str(e)))



if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="config.yaml")
    parsed_args=args.parse_args()
    data().clean_data(config_path=parsed_args.config)




