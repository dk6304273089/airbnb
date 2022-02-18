import os
from datetime import datetime
import yaml
import pandas as pd

import argparse
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

#reading paths in yaml file
def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
#logging the extract process
def log(file_object, log_message):
    now = datetime.now()
    date = now.date()
    current_time = now.strftime("%H:%M:%S")
    file_object.write(
        str(date) + "/" + str(current_time) + "\t\t" + log_message + "\n")
#extracting the data
class Data_extraction:
    def __init__(self):
        self.file = open("Training_logs/Training_log.txt", "a+")

    def get(self, config_path):
        try:
            config = read_params(config_path)
            database= config["database"]["data"]
            id=config["Credentials"]["Client_id"]
            secret=config["Credentials"]["Client_secret"]
            Data=config["Data"]["Raw"]
            cloud_config = {
                'secure_connect_bundle': "{}".format(database)}

            auth_provider = PlainTextAuthProvider(id,secret)
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider, idle_heartbeat_interval=25)
            session = cluster.connect()
            query="SELECT * FROM air.airbnb";
            log(self.file, "Data extraction from Astra DB has started")
            df = pd.DataFrame(list(session.execute(query)))
            df.to_csv(Data, index=False)
            log(self.file, "Data extraction from Astra DB has successfully completed")
        except Exception as e:
            log(self.file,e)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="config.yaml")
    parsed_args=args.parse_args()
    Data_extraction().get(config_path=parsed_args.config)