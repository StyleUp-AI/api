import re
import secrets
import jwt
import os
import uuid
import random
from string import Template
from azure.communication.email import EmailClient
from azure.storage.blob import BlobServiceClient
from flask import Blueprint, request, jsonify, make_response, redirect, url_for
from flask_cors import cross_origin
from datetime import datetime, timedelta
from src.main.routes import get_client, user_token_required, html_template, connection_string, azure_container_name, azure_email_connection_string, google_redirect_url
from werkzeug.security import check_password_hash, generate_password_hash
from google.oauth2 import id_token
from google.auth.transport import requests

user_routes = Blueprint("user_routes", __name__)


@user_routes.route("/log_event", methods=["POST"])
@cross_origin(origin='*')
def log_event():
    print(request.json)
    return make_response(jsonify({"data": "success"}), 200)

@user_routes.route("/signup", methods=["POST"])
@cross_origin(origin='*')
def signup():
    new_user = request.json
    db = get_client()
    users = db["users"]
    api_key = db["api_keys"]
   # otp = db["otp"]
   # if otp.find_one({"email": new_user["email"], "verified": "Yes"}) is None:
   #    return make_response(jsonify({"error": "Email: " + new_user["email"] + " not verified"}), 400)
    old_user = users.find_one({"email": new_user["email"]})
    if old_user:
        return make_response(
            jsonify({"error": "Email: " + new_user["email"] + " already exists"}), 400
        )
    hashed = generate_password_hash(new_user["password"])
    new_user_id = str(uuid.uuid4())
    users.insert_one(
        {"id": new_user_id, "email": new_user["email"], "password": hashed}
    )
    api_key.insert_one({"key": secrets.token_urlsafe(32), "user_id": new_user_id})
    return make_response(jsonify({"data": "User registeration success!"}), 201)

@user_routes.route("/get_otp", methods=["GET"])
@cross_origin(origin='*')
def get_otp():
    args = request.args
    num = random.randrange(1, 10**6)
    num_with_zeros = '{:06}'.format(num)
    if args["email"] is None:
        return make_response(jsonify({"error": "Must provide email"}), 400)
    db = get_client()
    otp = db["otp"]
    while otp.find_one({"code": num_with_zeros, "email": args["email"]}) is not None:
        num = random.randrange(1, 10**6)
        num_with_zeros = '{:06}'.format(num)
    otp.create_index("date", expireAfterSeconds=5*60)
    otp.insert_one({"code": num_with_zeros, "email": args["email"], "verified": "No"})

    azure_email_connection_string = os.environ.get("AZURE_EMAIL_CONNECTION_STRING")
    client = EmailClient.from_connection_string(azure_email_connection_string)
    message = {
        'content': {
            'subject': 'Your one time passcode',
            'plainText': 'Hi, Please find your otp: ' + str(num_with_zeros),
            'html': html_template.format(str(num_with_zeros))
        },
        'recipients': {
            'to': [
                {
                    'address': args["email"],
                    'displayName': 'Styleup AI'
                }
            ]
        },
        'senderAddress': 'noreply@styleup.fun'
    }
    poller = client.begin_send(message)
    print(poller.result())
    return make_response(jsonify({"data": "Otp sent"}), 200)



@user_routes.route("/verify_otp", methods=["POST"])
@cross_origin(origin='*')
def verify_otp():
    payload = request.json
    otp_code = payload['otp_code']
    email = payload['email']
    db = get_client()
    otp = db["otp"]
    if otp.find_one({'email': email, 'code': otp_code}) is None:
        return make_response(jsonify({"error": "Invalid otp"}), 400)
    otp.update_one({'email': email, 'code': otp_code}, {"$set": {"verified": "Yes"}})
    return make_response(jsonify({"data": "Otp verified"}), 200)

@user_routes.route("/reset_password", methods=["PATCH"])
@cross_origin(origin='*')
def reset_password():
    payload = request.json
    new_password = payload['password']
    confirm_password = payload['confirm_password']
    email = payload['email']
    db = get_client()
    users = db["users"]
    if new_password != confirm_password:
        return make_response(jsonify({"error": "Password not match"}), 400)
    pat = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?!.*[&%$]).{8,}$")
    if re.fullmatch(pat, new_password):
        users.update_one({"email": email}, {"$set":{"password": generate_password_hash(new_password)}})
        return make_response(jsonify({"data": "Password reset"}), 200)
    return make_response(jsonify({"error": "Password must be at least 8 characters and contain one upper case letter, one lower case letter, one number and one special character"}), 400)

@user_routes.route("/delete_user", methods=["DELETE"])
@cross_origin(origin='*')
@user_token_required
def delete_user(current_user):
    db = get_client()
    users = db["users"]
    api_key = db["api_keys"]
    for item in current_user["my_files"]:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service_client.get_container_client(azure_container_name)
            blob_client = container_client.get_blob_client(item)
            blob_client.delete_blob(delete_snapshots="include")
        except Exception as e:
            print(e)
    api_key.delete_many({"user_id": current_user["id"]})
    users.delete_one({"id": current_user["id"]})
    return make_response(jsonify({"data": "User delete successfully!"}), 200)

@user_routes.route("/add_api_key", methods=["POST"])
@cross_origin(origin='*')
@user_token_required
def add_api_key(current_user):
    db = get_client()
    api_keys = db["api_keys"]
    res = list(api_keys.find({"user_id": current_user['id']}))
    if len(res) > 3:
        return make_response(jsonify({"error": "One user can at most own 3 api keys"}), 400)
    api_keys.insert_one({"key": secrets.token_urlsafe(32), "user_id": current_user['id']})
    return make_response(jsonify({"data": "New API Key Added!"}), 201)

@user_routes.route("/get_api_keys", methods=["GET"])
@cross_origin(origin='*')
@user_token_required
def get_api_keys(current_user):
    db = get_client()
    api_keys = db["api_keys"]
    res = list(api_keys.find({"user_id": current_user['id']}))
    res = [r['key'] for r in res]
    return make_response(jsonify({"data": res}), 200)

@user_routes.route("/delete_api_key", methods=["DELETE"])
@cross_origin(origin='*')
@user_token_required
def delete_api_key(current_user):
    payload = request.json
    if 'key' not in payload or not payload['key']:
        return make_response(jsonify({"error": "Must provide api key"}), 400)
    db = get_client()
    api_keys = db["api_keys"]
    api_keys.delete_one({"key": payload['key'], 'user_id': current_user['id']})
    return make_response(jsonify({"data": 'Api key deleted'}), 200)

@user_routes.route("/google_sso", methods=["POST"])
@cross_origin(origin='*')
def google_sso():
    payload = request.json
    if 'token' not in payload or not payload['token']:
        return make_response(jsonify({"error": "Must provide token"}), 400)
    token = payload['token']
    CLIENT_ID = os.environ.get('GOOGLE_SSO_CLIENT_ID')
    idinfo = id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID, clock_skew_in_seconds=60)
    user_id = idinfo['sub']
    db = get_client()
    users = db["users"]
    api_key = db['api_keys']
    find_user = users.find_one({"id": user_id})
    if not find_user:
        users.insert_one(
            {"id": user_id, "email": payload["email"]}
        )
        api_key.insert_one({"key": secrets.token_urlsafe(32), "user_id": user_id})


    access_token = jwt.encode(
        {"user_id": user_id, "exp": datetime.utcnow() + timedelta(days=7)},
        os.environ.get("SECRET_KEY"),
        algorithm="HS256",
    )

    refresh_token = jwt.encode(
        {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(days=365),
        },
        os.environ.get("SECRET_KEY"),
        algorithm="HS256",
    )

    return make_response(
        jsonify(
            {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user_id": user_id,
            }
        ),
        200,
    )

@user_routes.route("/unsubscribe", methods=["GET"])
@cross_origin(origin='*')
def unsubscribe():
    args = request.args
    if not args or not args['id']:
        return redirect(url_for('.cannotunsubscribe', messages="Cannot unsubscribe"))
    db = get_client()
    early_schema = db["early_schema"]
    early_schema.update_one({'id': args['id']}, {"$set": {"want_email": "no"}})
    return make_response(jsonify({"data": "unsubscribe success!"}), 200)



@user_routes.route("/early_access", methods=["POST"])
@cross_origin(origin='*')
def early_access():
    payload = request.json
    if not payload or not payload['name'] or not payload['email'] or not payload['task']:
        return make_response(
            jsonify({"error": "Must provide user name, email, and task preference"}), 400, None
        )
    db = get_client()
    early_schema = db["early_schema"]
    find_early = early_schema.find_one({"email": payload['email']})
    if find_early:
        return make_response(
            jsonify({"error": "You have registered the early access with email: " + payload["email"]}),
            400,
        )
    new_id = str(uuid.uuid4())
    early_schema.insert_one({"id": new_id, "email": payload["email"], "task": payload['task'], "want_email": "yes"})
    early_email_template = open(os.path.join(os.getcwd(), 'src/main/constants/styleUp_email.html'), 'r', encoding='utf-8').read()
    client = EmailClient.from_connection_string(azure_email_connection_string)
    message = {
        'content': {
            'subject': 'Thank you for your interest in StyleUp',
            'plainText': "StyleUp AI, a no-code solution in which anyone could create, deploy, and manage large language model (LLM) agents in less than five minutes. StyleUp isn't just another simple AI builder but a comprehensive platform that allows you to connect and integrate your own private and unique dataset, creating AI agents tailored to your specific use cases and it’s coming soon! You’ll be one of the first to get early access in the next few days. As soon as the platform is ready for the official launch, you will be notified by email. \n\nBest regards,",
            'html': Template(early_email_template).safe_substitute(link = google_redirect_url + "/api/users/unsubscribe?id=" + new_id)
        },
        'recipients': {
            'to': [
                {
                    'address': payload["email"],
                    'displayName': 'Styleup AI'
                }
            ]
        },
        'senderAddress': 'noreply@styleup.fun'
    }
    poller = client.begin_send(message)
    return make_response(
        jsonify(
            {
                "data": "Early access registered",
            }
        ),
        200,
    )


@user_routes.route("/signin", methods=["POST"])
@cross_origin(origin='*')
def signin():
    payload = request.json
    if not payload or not payload["email"] or not payload["password"]:
        return make_response(
            jsonify({"error": "Must provide email and password"}), 400, None
        )
    db = get_client()
    users = db["users"]
    find_user = users.find_one({"email": payload["email"]})
    if not find_user:
        return make_response(
            jsonify({"error": "User not registered with email: " + payload["email"]}),
            401,
        )
    if check_password_hash(find_user["password"], payload["password"]):
        access_token = jwt.encode(
            {"user_id": find_user["id"], "exp": datetime.utcnow() + timedelta(days=7)},
            os.environ.get("SECRET_KEY"),
            algorithm="HS256",
        )

        refresh_token = jwt.encode(
            {
                "user_id": find_user["id"],
                "exp": datetime.utcnow() + timedelta(days=365),
            },
            os.environ.get("SECRET_KEY"),
            algorithm="HS256",
        )

        return make_response(
            jsonify(
                {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "user_id": find_user["id"],
                }
            ),
            200,
        )

    return make_response(jsonify({"error": "Could not verify"}), 403)
