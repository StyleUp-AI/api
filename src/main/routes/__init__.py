import certifi
import os
import jwt
from urllib.parse import quote_plus
from functools import wraps
from flask import Flask, request, jsonify, make_response

from pymongo.mongo_client import MongoClient

user_name = quote_plus(os.environ.get("USER_NAME"))
password = quote_plus(os.environ.get("PASSWORD"))
cluster = quote_plus(os.environ.get("CLUSTER"))
connection_string = os.environ.get("AZURE_CONNECTION_STRING")
azure_account_key = os.environ.get("AZURE_ACCOUNT_KEY")
azure_account_name = os.environ.get("AZURE_ACCOUNT_NAME")
azure_container_name = os.environ.get("AZURE_CONTAINER")

uri = (
    "mongodb+srv://"
    + user_name
    + ":"
    + password
    + "@"
    + cluster
    + "/?retryWrites=true&w=majority"
)

mongo_client = MongoClient(uri, tlsCAFile=certifi.where())

sk_prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if
it does not have an answer.

Chat:
{{$chat_history}}
User: {{$user_input}}
ChatBot: """.strip()

html_template = """\
<html>
  <body>
    <p>Hi,<br>
       Please find your otp {0}
    </p>
  </body>
</html>
"""

def get_client():
    return mongo_client["poke_chat"]


def user_token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "x-access-token" in request.headers:
            token = request.headers["x-access-token"]
        if not token:
            return jsonify({"message": "Token is missing !!"}), 401

        db = get_client()
        users = db["users"]
        try:
            data = jwt.decode(token, os.environ.get("SECRET_KEY"), algorithms=["HS256"])
            current_user = users.find_one({"id": data["user_id"]})
        except Exception as e:
            print(e)
            return jsonify({"message": "Token is invalid !!"}), 401
        return f(current_user, *args, **kwargs)

    return decorated


def bot_api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "bot-api-key" in request.headers:
            token = request.headers["bot-api-key"]
        if not token:
            return jsonify({"message": "Api Key is missing !!"}), 401

        db = get_client()
        keys = db["api_keys"]
        users = db["users"]
        try:
            api_key = keys.find_one({"key": key})
            current_user = users.find_one({"id": api_key["user_id"]})
        except Exception as e:
            print(e)
            return jsonify({"message": "API Key is invalid !!"}), 401
        return f(current_user, *args, **kwargs)

    return decorated
