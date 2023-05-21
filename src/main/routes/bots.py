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
from flask import Blueprint, request, jsonify, make_response
from typing import List
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer, util
from bson.objectid import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
from src.main.routes import user_token_required, bot_api_key_required, get_client, sk_prompt, s3, bucket_name, aws_domain


bots_routes = Blueprint("bots_routes", __name__)
@bots_routes.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    # Other headers can be added here if needed
    return response

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
pool = concurrent.futures.ThreadPoolExecutor()


def reset_context(relevance_score):
    global context
    global chat_func
    global kernel

    context = kernel.create_new_context()
    context["chat_history"] = ""

    chat_func = kernel.create_semantic_function(
        sk_prompt, max_tokens=200, temperature=relevance_score
    )


def add_collection_from_file(file, file_name, file_extension):
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
    json_file_name = "Documents/" + current_user["id"] + "/" + file_name + ".json"
    try:
        s3.put_object(Bucket=bucket_name, Key=file_name, Body=json.dumps(res, indent=2, default=str))
    except Exception as e:
        print(e)
        raise
    if "my_files" in current_user:
        current_user["my_files"].append(file_name)
    else:
        current_user["my_files"] = [file_name]
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

    data = {}
    with open(file_name) as j:
        data = json.load(j)
    vectors_input = model.encode(user_input)
    res = cosine_similarity([vectors_input], data["vectors"])
    max_index = np.argmax(res)
    max_value = np.max(res)
    if max_value >= relevance_score:
        return data["texts"][max_index]

    default_answer = await kernel.run_async(chat_func, input_vars=context.variables)
    context["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {default_answer}\n"

    return default_answer.result


@bots_routes.route("/update_collection", methods=["PUT"])
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
    file_name = (
        "Documents/" + current_user["id"] + "/" + payload["collection_name"] + ".json"
    )
    try:
        s3.put_object(Bucket=bucket_name, Key=file_name, Body=json.dumps(res, indent=2, default=str))
    except Exception as e:
        print(e)
        return make_response(jsonify({"error": "Cannot update the collection"}), 400)
    return make_response(jsonify({"data": "Collection updated"}), 200)


@bots_routes.route("/get_collections", methods=["GET"])
@user_token_required
def get_collections(current_user):
    return make_response(jsonify({"data": current_user["my_files"]}), 200)


@bots_routes.route("/get_collection", methods=["GET"])
@user_token_required
def get_collection(current_user):
    args = request.args
    if args["collection_name"] is None:
        return make_response(jsonify({"error": "Must provide collection name"}), 400)
    file_name = (
        "Documents/" + current_user["id"] + "/" + args["collection_name"] + ".json"
    )
    res = s3.get_object(Bucket=bucket_name, Key=file_name)
    data = json.loads(res['Body'].read().decode("utf-8"))
    return make_response(jsonify({"data": data['texts']}), 200)


@bots_routes.route("/delete_collection", methods=["DELETE"])
@user_token_required
def delete_collection(current_user):
    args = request.args
    if args["collection_name"] is None:
        return make_response(jsonify({"error": "Must provide collection name"}), 400)

    file_name = (
        "Documents/" + current_user["id"] + "/" + args["collection_name"] + ".json"
    )
    if file_name not in current_user["my_files"]:
        return make_response(
            jsonify({"error": "User does not own this collection"}), 400
        )

    try:
        s3.Object(bucket_name, file_name).delete()
    except Exception as e:
        print(e)
        return make_response(jsonify({"error": "Cannot remove collection"}), 400)

    db = get_client()
    users = db["users"]
    current_user["my_files"].remove(file_name)
    users.update_one({'id': current_user['id']}, {'$set': {'my_files': current_user['my_files']}})
    return make_response(jsonify({"data": "File removed successfully"}), 200)


@bots_routes.route("/add_collection", methods=["POST"])
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
    file_name = (
        "Documents/" + current_user["id"] + "/" + payload["collection_name"] + ".json"
    )
    try:
        s3.put_object(Bucket=bucket_name, Key=file_name, Body=json.dumps(res, indent=2, default=str))
    except Exception as e:
        print(e)
        return make_response(jsonify({"error": "Cannot save the collection"}), 400)
    db = get_client()
    users = db["users"]
    if "my_files" in current_user:
        current_user["my_files"].append(file_name)
    else:
        current_user["my_files"] = [file_name]
    users.update_one(
        {"id": current_user["id"]}, {"$set": {"my_files": current_user["my_files"]}}
    )
    return make_response(jsonify({"data": "New collection added"}), 201)


@bots_routes.route("/add_collection_batch", methods=["POST"])
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
        add_collection_from_file(file, file_name, file_extension)
    return make_response(jsonify({"data": "New collection added"}), 201)


@bots_routes.route("/reset_context", methods=["POST"])
@bot_api_key_required
def reset_context(current_user):
    payload = request.json
    reset_context(payload["relevance_score"])
    return make_response(jsonify({"message": "Context refreshed"}), 200)


@bots_routes.route("/chat", methods=["POST"])
@bot_api_key_required
def chat(current_user):
    payload = request.json
    if payload is None or payload["input"] is None:
        return make_response(jsonify({"error": "Must provide input key"}), 400)
    file_name = "Documents/" + current_user["id"] + "/" + payload["collection_name"]
    if file_name not in current_user["my_files"]:
        return make_response(jsonify({"error": "User does't have this file"}), 400)
    result = pool.submit(
        asyncio.run, talk_bot(payload["input"], file_name, payload["relevance_score"])
    ).result()
    return make_response(jsonify({"data": result}), 200)
