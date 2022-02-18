import cassandra
import pandas as pd

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
cloud_config= {
        'secure_connect_bundle': r'E:/secure-connect-airbnb.zip'
}
auth_provider = PlainTextAuthProvider('GUIuEtGFJZyhLRdIyFaymbvE', 'xE2XLTL32wBenKZtBctt8Moym-tLbyuKU0_zWkYIRE0XwlKNqi+jNdE8QK21Q0nFQBi76a6.NiA+ibFu81G53-iBKuHKNjGCAjZ8ATbdymaP3iCO1J0N__M2Re,7RMEo')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()
query = "SELECT * FROM air.airbnb";
df1 = pd.DataFrame(list(session.execute(query)))
print(df1.head())