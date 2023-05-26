import uuid
import os
import json
import csv
import pandas as pd
import numpy as np
import asyncio, concurrent.futures
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    OpenAITextCompletion,
    OpenAITextEmbedding,
)
from azure.storage.blob import BlobServiceClient
from flask import Blueprint, request, jsonify, make_response
from flask_cors import cross_origin
from typing import List
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer, util
from bson.objectid import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
from src.main.routes import user_token_required, bot_api_key_required, get_client, sk_prompt, connection_string, azure_container_name
from src.main.utils import train_ml_model

bots_routes = Blueprint("bots_routes", __name__)
@bots_routes.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,PATCH,POST,DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept'
    # Other headers can be added here if needed
    return response

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
pool = concurrent.futures.ThreadPoolExecutor()

kernel = sk.Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
api_key = os.environ.get("OPENAI_API_KEY")
org_id = os.environ.get("OPENAI_ORG_ID")
kernel.add_text_completion_service(
    "dv", OpenAITextCompletion("text-davinci-003", api_key, org_id)
)
kernel.add_text_embedding_generation_service(
    "ada", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id)
)
kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())
kernel.import_skill(sk.core_skills.TextMemorySkill())

context = kernel.create_new_context()
context["chat_history"] = ""

chat_func = kernel.create_semantic_function(sk_prompt, max_tokens=200, temperature=0.8)

def upload_to_blob_storage(file_path, file_name, data):
    destination = file_path + '/' + file_name
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=azure_container_name, blob=destination)
    blob_client.upload_blob(data, overwrite=True)

def reset_context(relevance_score):
    global context
    global chat_func
    global kernel

    context = kernel.create_new_context()
    context["chat_history"] = ""

    chat_func = kernel.create_semantic_function(
        sk_prompt, max_tokens=200, temperature=relevance_score
    )

def update_collection_from_file(file, file_name, file_extension, old_collection, current_user):
    file_path = "Tmp/" + current_user["id"] + "/" + file.filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)
    data = []
    if file_extension == "csv":
        with open(file_path) as f:
            csv_file = csv.reader(f)
            for row in csv_file:
                data.append(row.split(","))
    else:
        with open(file_path) as f:
            excel_file = pd.read_excel(f, header=None)
            for column in excel_file.columns:
                column_values = excel_file[column].tolist()
                data.append(column_values)

    res = {}
    vectors = model.encode(data)
    arr = np.array(vectors)
    arr_as_list = arr.tolist()
    res["vectors"] = arr_as_list
    res["texts"] = data
    json_file_path = "Documents/" + current_user["id"]
    json_file_name = old_collection + ".json"
    try:
        upload_to_blob_storage(json_file_path, json_file_name, json.dumps(res, indent=2, default=str))
    except Exception as e:
        print(e)
        raise

    users.update_one(
        {"id": current_user["id"]}, {"$set": {"my_files": current_user["my_files"]}}
    )
    os.remove(file_path)


def add_collection_from_file(file, file_name, file_extension, current_user):
    file_path = "Tmp/" + current_user["id"] + "/" + file.filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)
    data = []
    if file_extension == "csv":
        with open(file_path) as f:
            csv_file = csv.reader(f)
            for row in csv_file:
                data.append(row.split(","))
    else:
        with open(file_path) as f:
            excel_file = pd.read_excel(f, header=None)
            for column in excel_file.columns:
                column_values = excel_file[column].tolist()
                data.append(column_values)

    res = {}
    vectors = model.encode(data)
    arr = np.array(vectors)
    arr_as_list = arr.tolist()
    res["vectors"] = arr_as_list
    res["texts"] = data
    json_file_path = "Documents/" + current_user["id"]
    json_file_name = file_name + ".json"
    try:
        upload_to_blob_storage(json_file_path, json_file_name, json.dumps(res, indent=2, default=str))
    except Exception as e:
        print(e)
        raise
    if "my_files" in current_user:
        current_user["my_files"].append(json_file_name)
    else:
        current_user["my_files"] = [json_file_name]
    users.update_one(
        {"id": current_user["id"]}, {"$set": {"my_files": current_user["my_files"]}}
    )
    os.remove(file_path)


async def talk_bot(user_input, file_name, relevance_score):
    global context
    global chat_func
    global kernel
    try:
        context["user_input"] = user_input
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return ""
    except EOFError:
        print("\n\nExiting chat...")
        return ""

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(azure_container_name)
    blob_client = container_client.get_blob_client(file_name)
    res = blob_client.download_blob().readall()
    data = json.loads(res)
    vectors_input = model.encode(user_input)
    res = cosine_similarity([vectors_input], data["vectors"])
    max_index = np.argmax(res)
    max_value = np.max(res)
    if max_value >= relevance_score:
        bot_answer = data["texts"][max_index]
        context["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {bot_answer}\n"
        return data["texts"][max_index]

    default_answer = await kernel.run_async(chat_func, input_context=context)
    context["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {default_answer}\n"

    return default_answer.result


@bots_routes.route("/update_collection", methods=["PUT"])
@cross_origin(origin='*')
@user_token_required
def update_collection(current_user):
    payload = request.json
    if payload["collection_list"] is None or payload["collection_name"] is None:
        return make_response(
            jsonify({"error": "Collection items must not be none"}), 400
        )
    res = {}
    vectors = model.encode(payload["collection_list"])
    arr = np.array(vectors)
    arr_as_list = arr.tolist()
    res["vectors"] = arr_as_list
    res["texts"] = payload["collection_list"]
    file_path = "Documents/" + current_user["id"]
    file_name = payload["collection_name"] + ".json"
    try:
        upload_to_blob_storage(file_path, file_name, json.dumps(res, indent=2, default=str))
    except Exception as e:
        print(e)
        return make_response(jsonify({"error": "Cannot update the collection"}), 400)
    return make_response(jsonify({"data": "Collection updated"}), 200)


@bots_routes.route("/get_collections", methods=["GET"])
@cross_origin(origin='*')
@user_token_required
def get_collections(current_user):
    return make_response(jsonify({"data": current_user["my_files"]}), 200)


@bots_routes.route("/get_collection", methods=["GET"])
@cross_origin(origin='*')
@user_token_required
def get_collection(current_user):
    args = request.args
    if args["collection_name"] is None:
        return make_response(jsonify({"error": "Must provide collection name"}), 400)
    if args["collection_name"] + '.json' not in current_user['my_files']:
        return make_response(jsonify({"error": "User doens't own this collection"}), 400)
    file_path = "Documents/" + current_user["id"]
    file_name = args["collection_name"] + ".json"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(azure_container_name)
    blob_client = container_client.get_blob_client(file_path + '/' + file_name)
    res = blob_client.download_blob().readall()
    data = json.loads(res)
    return make_response(jsonify({"data": data['texts']}), 200)


@bots_routes.route("/delete_collection", methods=["DELETE"])
@cross_origin(origin='*')
@user_token_required
def delete_collection(current_user):
    args = request.args
    if args["collection_name"] is None:
        return make_response(jsonify({"error": "Must provide collection name"}), 400)
    file_path = "Documents/" + current_user["id"]
    file_name = args["collection_name"] + ".json"
    if file_name not in current_user["my_files"]:
        return make_response(
            jsonify({"error": "User does not own this collection"}), 400
        )

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(azure_container_name)
        blob_client = container_client.get_blob_client(file_path + '/' + file_name)
        blob_client.delete_blob(delete_snapshots="include")
    except Exception as e:
        print(e)
        return make_response(jsonify({"error": "Cannot remove collection"}), 400)

    db = get_client()
    users = db["users"]
    current_user["my_files"].remove(file_name)
    users.update_one({'id': current_user['id']}, {'$set': {'my_files': current_user['my_files']}})
    return make_response(jsonify({"data": "File removed successfully"}), 200)

@bots_routes.route("/train_collection", methods=["POST"])
@cross_origin(origin='*')
@user_token_required
def train_collection(current_user):
    payload = request.json
    if payload["collection_name"] is None:
        return make_response(
            jsonify({"error": "Collection name must not be none"}), 400
        )
    if payload["collection_name"] + '.json' not in current_user['my_files']:
        return make_response(jsonify({"error": "User doens't own this collection"}), 400)
    file_path = "Documents/" + current_user["id"]
    file_name = payload["collection_name"] + ".json"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(azure_container_name)
    blob_client = container_client.get_blob_client(file_path + '/' + file_name)
    res = blob_client.download_blob().readall()
    data = json.loads(res)
    res = train_ml_model(data['texts'])
    print(res)
    return make_response(jsonify({"data": res}), 200)
    

@bots_routes.route("/add_collection", methods=["POST"])
@cross_origin(origin='*')
@user_token_required
def add_collection(current_user):
    payload = request.json
    if payload["collection_list"] is None:
        return make_response(
            jsonify({"error": "Collection items must not be none"}), 400
        )
    res = {}
    vectors = model.encode(payload["collection_list"])
    arr = np.array(vectors)
    arr_as_list = arr.tolist()
    res["vectors"] = arr_as_list
    res["texts"] = payload["collection_list"]
    json_file_name = payload["collection_name"] + ".json"
    json_file_path = "Documents/" + current_user["id"]
    if json_file_name in current_user['my_files']:
        return make_response(
            jsonify({"error": "Collection name must be unique"}), 400
        )

    try:
        upload_to_blob_storage(json_file_path, json_file_name, json.dumps(res, indent=2, default=str))
    except Exception as e:
        print(e)
        return make_response(jsonify({"error": "Cannot save the collection"}), 400)
    db = get_client()
    users = db["users"]
    if "my_files" in current_user:
        current_user["my_files"].append(json_file_name)
    else:
        current_user["my_files"] = [json_file_name]
    users.update_one(
        {"id": current_user["id"]}, {"$set": {"my_files": current_user["my_files"]}}
    )
    return make_response(jsonify({"data": "New collection added"}), 201)

@bots_routes.route("/update_collection_batch", methods=["PUT"])
@cross_origin(origin='*')
@user_token_required
def update_collection_batch(current_user):
    payload = request.json
    if "files" not in request.files:
        return make_response(jsonify({"error": "File must provide"}), 400)
    files = request.files["files"]
    old_collection = payload['collection_name']
    db = get_client()
    users = db["users"]
    for i, file in enumerate(files):
        (file_name, file_extension) = os.path.splitext(file.filename)
        if file_extension != "csv" or file_extension != "xlsx":
            return make_response(
                jsonify(
                    {"error: " "File: " + file_name + " is not valid csv or excel file"}
                ),
                400,
            )
        update_collection_from_file(file, file_name, file_extension, old_collection, current_user)
    return make_response(jsonify({"data": "Update collection success!"}), 200)


@bots_routes.route("/add_collection_batch", methods=["POST"])
@cross_origin(origin='*')
@user_token_required
def add_collection_batch(current_user):
    payload = request.json
    if "files" not in request.files:
        return make_response(jsonify({"error": "File must provide"}), 400)
    files = request.files["files"]
    db = get_client()
    users = db["users"]
    for i, file in enumerate(files):
        (file_name, file_extension) = os.path.splitext(file.filename)
        if file_extension != "csv" or file_extension != "xlsx":
            return make_response(
                jsonify(
                    {"error: " "File: " + file_name + " is not valid csv or excel file"}
                ),
                400,
            )
        add_collection_from_file(file, file_name, file_extension, current_user)
    return make_response(jsonify({"data": "New collection added"}), 201)


@bots_routes.route("/reset_context", methods=["POST"])
@cross_origin(origin='*')
@bot_api_key_required
def reset_context(current_user):
    payload = request.json
    reset_context(payload["relevance_score"])
    return make_response(jsonify({"message": "Context refreshed"}), 200)


@bots_routes.route("/chat", methods=["POST"])
@cross_origin(origin='*')
@bot_api_key_required
def chat(current_user):
    payload = request.json
    if payload is None or payload["input"] is None:
        return make_response(jsonify({"error": "Must provide input key"}), 400)
    file_name = payload["collection_name"] + ".json"
    if file_name not in current_user["my_files"]:
        return make_response(jsonify({"error": "User does't have this file"}), 400)
    result = pool.submit(
        asyncio.run, talk_bot(payload["input"], file_name, payload["relevance_score"])
    ).result()
    #result = await talk_bot(payload["input"], file_name, payload["relevance_score"])
    return make_response(jsonify({"data": result}), 200)
