from extract_data import read_params,log
import pandas as pd
import re
import argparse
include_cols=["city","longitude","latitude","review_scores_rating","number_of_reviews","minimum_nights","accommodates","bathrooms","bedrooms","beds","security_deposit","cleaning_fee","property_type","room_type","availability_365","host_identity_verified","host_is_superhost","cancellation_policy","price","zipcode","state"]
price_list = ["price","cleaning_fee","security_deposit"]
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
            config = read_params(config_path)
            data=config["Data"]["Processed"]
            df = get_data(config_path)
            df=df[include_cols]
            log(self.file, "successfully Removed unwanted columns")
            for col in price_list:
                df[col] = df[col].fillna("0")
                df[col] = df[col].apply(lambda x: float(re.compile('[^0-9eE.]').sub('', x)) if len(x) > 0 else 0)
            df = df[df["price"] < 4000]
            df["city"] = df["city"].replace(
            {"Bondi": "Bondi Beach", "North Bondi": "Bondi Beach", "Bondi Junction": "Bondi Beach"})
            df["host_identity_verified"] = df["host_identity_verified"].map({"f": 0, "t": 1})
            df["host_is_superhost"] = df["host_is_superhost"].map({"f": 0, "t": 1})
            df.to_csv(data, index=False)
            log(self.file, "successfully Done Data cleaning and saved the file in {}".format(data))
        except Exception as e:
            log(self.file,str(e))


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="config.yaml")
    parsed_args=args.parse_args()
    data().clean_data(config_path=parsed_args.config)




