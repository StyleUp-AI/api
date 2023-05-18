import uuid
import os
import json
import numpy as np
import asyncio, concurrent.futures
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAITextCompletion, OpenAITextEmbedding
from flask import Blueprint, request, jsonify, make_response
from typing import List
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer, util
from bson.objectid import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
from src.main.routes import user_token_required, bot_api_key_required, get_client

bots_routes = Blueprint('bots_routes', __name__)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

kernel = sk.Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
api_key = os.environ.get("OPENAI_API_KEY")
org_id = os.environ.get("OPENAI_ORG_ID")
kernel.add_text_completion_service("dv", OpenAITextCompletion("text-davinci-003", api_key, org_id))
kernel.add_text_embedding_generation_service("ada", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id))
kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())
kernel.import_skill(sk.core_skills.TextMemorySkill())

sk_prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if
it does not have an answer.

Chat:
{{$chat_history}}
User: {{$user_input}}
ChatBot: """.strip()

context = kernel.create_new_context()
context["chat_history"] = ""

chat_func = kernel.create_semantic_function(sk_prompt, max_tokens=200, temperature=0.8)
pool = concurrent.futures.ThreadPoolExecutor()

def setup_chat_with_memory(relevance_score: float) -> None:
    global context
    global chat_func
    global kernel
    sk_prompt = """
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say 'I don't know' if
    it does not have an answer.

    Chat:
    {{$chat_history}}
    User: {{$user_input}}
    ChatBot: """.strip()

    context = kernel.create_new_context()
    context["chat_history"] = ""

    chat_func = kernel.create_semantic_function(sk_prompt, max_tokens=200, temperature=relevance_score)

async def talk_bot(user_input: str, file_name: str, relevance_score: float) -> dict:
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
    res = cosine_similarity([vectors_input], data['vectors'])
    max_index = np.argmax(res)
    max_value = np.max(res)
    if max_value >= relevance_score:
        return data['texts'][max_index]

    default_answer = await kernel.run_async(chat_func, input_vars=context.variables)
    context["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {default_answer}\n"

    return default_answer.result

@bots_routes.route('/add_collection', methods=['POST'])
@user_token_required
def add_collection(current_user):
    payload = request.json
    if payload['collection_list'] is None:
        return make_response(jsonify({'error': 'Collection items must not be none'}), 400)
    res = {}
    code_list = []
    vectors = model.encode(payload['collection_list'])
    arr = np.array(vectors)
    arr_as_list = arr.tolist()
    res['vectors'] = arr_as_list
    res['texts'] = payload['collection_list']
    current_date_time = datetime.now(timezone.utc)
    file_name = 'Documents/' + current_user['id'] + '/' + payload['collection_name'] + '.json'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w+') as f:
        json.dump(res, f)
    db = get_client()
    users = db['users']
    if 'my_files' in current_user:
        current_user['my_files'].append(file_name)
    else:
        current_user['my_files'] = [file_name]
    users.update_one({'id': current_user['id']}, {'$set':{'my_files': current_user['my_files']}})
    return make_response(jsonify({'data': 'New collection added'}), 201)

@bots_routes.route('/add_collection_batch', methods=['POST'])
@user_token_required
def add_collection_batch(current_user):
    payload = request.json
    if payload['collection_list'] is None:
        return make_response(jsonify({'error': 'Collection items must not be none'}), 400)
    res = {}
    code_list = []
    vectors = model.encode(payload['collection_list'])
    arr = np.array(vectors)
    arr_as_list = arr.tolist()
    res['vectors'] = arr_as_list
    res['texts'] = payload['collection_list']
    current_date_time = datetime.now(timezone.utc)
    file_name = 'Documents/' + current_user['id'] + '/' + payload['collection_name'] + '.json'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w+') as f:
        json.dump(res, f)
    db = get_client()
    users = db['users']
    if 'my_files' in current_user:
        current_user['my_files'].append(file_name)
    else:
        current_user['my_files'] = [file_name]
    users.update_one({'id': current_user['id']}, {'$set':{'my_files': current_user['my_files']}})
    return make_response(jsonify({'data': 'New collection added'}), 201)

@bots_routes.route('/reset_context', methods=['POST'])
@bot_api_key_required
def reset_context(current_user):
    payload = request.json
    setup_chat_with_memory(payload['relevance_score'])
    return make_response(jsonify({'message': 'Context refreshed'}), 200)

@bots_routes.route('/chat', methods=['POST'])
@bot_api_key_required
def chat(current_user):
    payload = request.json
    if payload is None or payload['input'] is None:
        return make_response(jsonify({'error': 'Must provide input key'}), 400)
    file_name = 'Documents/' + current_user['id'] + '/' + payload['collection_name']
    if file_name not in current_user['my_files']:
        return make_response(jsonify({'error': 'User does\'t have this file'}), 400)
    result = pool.submit(asyncio.run, talk_bot(payload['input'], file_name, payload['relevance_score'])).result()
    return make_response(jsonify({'data': result}), 200)
