import json
from getpass import getpass
import pyrebase
from django.contrib.sites import requests

firebaseConfig = {
    "apiKey": "AIzaSyA-saCFmPsUdolkqVKo99btN2-1kmpOc0g",
    "authDomain": "heartdiseaseprediction-3fe1a.firebaseapp.com",
    "projectId": "heartdiseaseprediction-3fe1a",
    "databaseURL": "xxxxxx",
    "storageBucket": "heartdiseaseprediction-3fe1a.appspot.com",
    "messagingSenderId": "238231577148",
    "appId": "1:238231577148:web:d4dacca7602cb3d1e9d939",
    "measurementId": "G-E0210YFMK3"
}
import requests.exceptions

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

email = input("Please Enter Your Email Address : \n")
password = getpass("Please Enter Your Password : \n")

try:  # create users
    user = auth.create_user_with_email_and_password(email, password)
    print("Success .... ")
except requests.exceptions.HTTPError as httpErr:
    error_message = json.loads(httpErr.args[1])['error']['message']
    print(error_message)

# auth.sign_in_with_email_and_password(email, password)

# send email verification
# auth.send_email_verification(login['idToken'])

# reset the password
auth.send_password_reset_email(email)

print("Success ... ")
import requests.exceptions  # error types
