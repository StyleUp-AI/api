from flask import Blueprint, request, jsonify, make_response
from datetime import datetime, timedelta
from src.main.routes import get_client, user_token_required
from werkzeug.security import check_password_hash, generate_password_hash
import secrets
import jwt
import os
import uuid

user_routes = Blueprint('user_routes', __name__)

@user_routes.route('/signup', methods=['POST'])
def signup():
    new_user = request.json
    db = get_client()
    users = db['users']
    api_key = db['api_keys']
    old_user = users.find_one({'email': new_user['email']})
    if old_user:
        return make_response(jsonify({'error': 'Email: ' + new_user['email'] + ' already exists'}), 400)
    hashed = generate_password_hash(new_user['password'])
    new_user_id = str(uuid.uuid4())
    users.insert_one({'id': new_user_id,'email': new_user['email'], 'password': hashed})
    api_key.insert_one({'key': secrets.token_urlsafe(32),'user_id': new_user_id})
    return make_response(jsonify({'data': 'User registeration success!'}), 201)

@user_routes.route('/delete_user', methods=['DELETE'])
@user_token_required
def delete_user(current_user):
    db = get_client()
    users = db['users']
    api_key = db['api_keys']
    for item in current_user['my_files']:
        os.remove(item):
    api_keys.delete_many({'user_id': current_user['id']})
    users.delete_one({'id': current_user['id']})
    return make_response(jsonify({'data': 'User delete successfully!'}), 200)


@user_routes.route('/signin', methods=['POST'])
def signin():
    payload = request.json
    if not payload or not payload['email'] or not payload['password']:
        return make_response(jsonify({'error': 'Must provide email and password'}), 400, None)
    db = get_client()
    users = db['users']
    find_user = users.find_one({'email': payload['email']})
    if not find_user:
        return make_response(jsonify({'error': 'User not registered with email: ' + payload['email']}), 401)
    if check_password_hash(find_user['password'], payload['password']):
        access_token = jwt.encode({
            'user_id': find_user['id'],
            'exp' : datetime.utcnow() + timedelta(days = 7)
        }, os.environ.get("SECRET_KEY"), algorithm="HS256")

        refresh_token = jwt.encode({
            'user_id': find_user['id'],
            'exp' : datetime.utcnow() + timedelta(days = 365)
        }, os.environ.get("SECRET_KEY"), algorithm="HS256")

        return make_response(jsonify({'access_token' : access_token, 'refresh_token': refresh_token, 'user_id': find_user['id']}), 200)

    return make_response(
        jsonify({'error': 'Could not verify'}),
        403
    )
