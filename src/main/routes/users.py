import re
import secrets
import jwt
import os
import uuid
import random
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Blueprint, request, jsonify, make_response
from datetime import datetime, timedelta
from src.main.routes import get_client, user_token_required, html_template
from werkzeug.security import check_password_hash, generate_password_hash

user_routes = Blueprint("user_routes", __name__)


@user_routes.route("/signup", methods=["POST"])
def signup():
    new_user = request.json
    db = get_client()
    users = db["users"]
    api_key = db["api_keys"]
    otp = db["otp"]
    if otp.find_one({"email": new_user["email"], "verified": "Yes"}) is None:
        return make_response(jsonify({"error": "Email: " + new_user["email"] + " not verified"}), 400)
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
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    message = MIMEMultipart("alternative")
    message['Subject'] = 'Your one time code'
    message['From'] = sender_email
    message['To'] = args['email']
    html_template.format(str(num_with_zeros))
    part = MIMEText(html_template, "html")
    message.attach(part)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, sender_password)
        server.sendmail(
            sender_email, args["email"], message.as_string()
        )
    otp.ensure_index("date", expireAfterSeconds=5*60)
    otp.insert_one({"code": num_with_zeros, "email": args["email"], "verified": "No"})
    return make_response(jsonify({"data": "Otp sent"}), 200)

@user_routes.route("/verify_otp", methods=["POST"])
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
@user_token_required
def delete_user(current_user):
    db = get_client()
    users = db["users"]
    api_key = db["api_keys"]
    for item in current_user["my_files"]:
        try:
            s3.Object(bucket_name, item).delete()
        except Exception as e:
            print(e)
            return make_response(jsonify({"error": "Cannot delete user"}), 400)
    api_keys.delete_many({"user_id": current_user["id"]})
    users.delete_one({"id": current_user["id"]})
    return make_response(jsonify({"data": "User delete successfully!"}), 200)


@user_routes.route("/signin", methods=["POST"])
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
