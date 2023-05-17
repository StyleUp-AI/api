import uuid
import os
import asyncio, concurrent.futures
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAITextCompletion, OpenAITextEmbedding, AzureTextCompletion
from flask import Blueprint, request, jsonify, make_response
from typing import List
import semantic_kernel as sk
from src.main.routes import user_token_required, bot_api_key_required, get_client

bots_routes = Blueprint('bots_routes', __name__)


kernel = sk.Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
api_key = os.environ.get("OPENAI_API_KEY")
org_id = os.environ.get("OPENAI_ORG_ID")
kernel.add_text_completion_service("dv", OpenAITextCompletion("text-davinci-003", api_key, org_id))
kernel.add_text_embedding_generation_service("ada", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id))
kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())
kernel.import_skill(sk.core_skills.TextMemorySkill())

context = None
pool = concurrent.futures.ThreadPoolExecutor()
chat_func = None

async def populate_memory(collection_param: str, bot_answers: List[str]) -> None:
    # Add some documents to the semantic memory
    global kernel
    for index, item in enumerate(bot_answers):
        await kernel.memory.save_information_async(collection_param, item['text'], item['id'])

def setup_chat_with_memory(collection_param: str, relevance_score: float) -> None:
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

    context[sk.core_skills.TextMemorySkill.COLLECTION_PARAM] = collection_param
    context[sk.core_skills.TextMemorySkill.RELEVANCE_PARAM] = relevance_score

async def talk_bot(user_input: str, collection_name: str, relevance_score: float) -> dict:
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

    result = await kernel.memory.search_async(collection_name, user_input)
    if result[0].relevance >= relevance_score:
        context["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {result[0].text}\n"
        return result[0].text

    default_answer = await kernel.run_async(chat_func, input_vars=context.variables)
    context["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {default_answer}\n"

    return default_answer.result

@bots_routes.route('/add_collection', methods=['POST'])
@user_token_required
def add_collection(current_user):
    payload = request.json
    db = get_client()
    bots = db['bots']
    bots.insert_one({'id': str(uuid.uuid4()),
        'user_id': current_user['id'],
        'collection_name': payload['collection_name'],
        'collection_list': payload['collection_list'],
        'relevance_score': payload['relevance_score']})
    return make_response(jsonify({'data': 'New db collection added'}), 201)

@bots_routes.route('/populate_kernel', methods=['POST'])
@user_token_required
def populate_kernel(current_user):
    payload = request.json
    db = get_client()
    bots = db['bots']
    find_bot = bots.find_one({'user_id': current_user['id'], 'id': payload['collection_id']})
    if not find_bot:
        return make_response(jsonify({'error': 'Bot not found'}), 400)
    setup_chat_with_memory(find_bot['collection_name'], find_bot['relevance_score'])
    pool.submit(asyncio.run, populate_memory(find_bot['collection_name'], find_bot['collection_list'])).result()
    return make_response(jsonify({'collection_name': find_bot['collection_name'], 'relevance_score': find_bot['relevance_score'], 'message': 'Kernel populated successfully'}), 200)

@bots_routes.route('/chat', methods=['POST'])
@user_token_required
def chat(current_user):
    payload = request.json
    if payload is None or payload['input'] is None:
        return make_response(jsonify({'error': 'Must provide input key'}), 400)
    result = pool.submit(asyncio.run, talk_bot(payload['input'], payload['collection_name'], payload['relevance_score'])).result()
    return make_response(jsonify({'data': result}), 200)
