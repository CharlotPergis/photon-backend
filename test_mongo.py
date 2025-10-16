from pymongo import MongoClient

# Replace with your actual connection string
uri = "mongodb+srv://pergishazel_db_user:6hgkaPrWF9Q5wPpF@photoncluster.wa6xetd.mongodb.net/?retryWrites=true&w=majority&appName=PhotonCluster"

# Connect using modern pymongo options
client = MongoClient(uri, tls=True, tlsAllowInvalidCertificates=False)

try:
    # Test connection
    print(client.server_info())  # Should print server info if successful
except Exception as e:
    print("Connection failed:", e)
