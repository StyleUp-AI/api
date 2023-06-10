import asyncio
import os
import json
import urllib
import asyncio, concurrent.futures
from multiprocessing import Process
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from flask import Blueprint, request, jsonify, make_response
from flask_cors import cross_origin
from sentence_transformers import SentenceTransformer
from src.main.routes import user_token_required, bot_api_key_required, get_client, sk_prompt, connection_string, azure_container_name, user_sessions
from src.main.utils.model_actions import train_mode

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

# Prepare OpenAI service using credentials stored in the `.env` file
api_key = os.environ.get("OPENAI_API_KEY")
org_id = os.environ.get("OPENAI_ORG_ID")

def upload_to_blob_storage(file_path, file_name, data):
    destination = file_path + '/' + file_name
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=azure_container_name, blob=destination)
    blob_client.upload_blob(data, overwrite=True)

def reset_context_helper(current_user):
    global user_sessions
    global sk_prompt
    user_sessions[current_user['id']] = {
        'context': ConversationBufferMemory(return_messages=True),
        'prompt_template': sk_prompt
    }

async def talk_bot(user_input, file_name, current_user, relevance_score):
    global user_sessions
    global llm
    if current_user['id'] not in user_sessions:
        reset_context_helper(current_user)
    conversation_memory = user_sessions[current_user['id']]['context']
    az_loaders = AzureBlobStorageFileLoader(connection_string, azure_container_name, file_name)
    loaders = az_loaders.load()
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_rerank", return_intermediate_steps=True)
    results = chain({"input_documents": loaders, "question": user_input}, return_only_outputs=True)
    results = results["intermediate_steps"]
    max_score_item = max(results, key=lambda x:float(x['score']))
    print(max_score_item)
    if float(max_score_item['score']) >= relevance_score:
        conversation_memory.chat_memory.add_user_message(user_input)
        conversation_memory.chat_memory.add_ai_message(max_score_item['answer'])
        return max_score_item['answer']
    conversation = ConversationChain(memory=conversation_memory, prompt=user_sessions[current_user['id']]['prompt_template'], llm=OpenAI(temperature=0))
    obj = conversation.predict(input=user_input).split("AI: ",1)[1]
    print(obj)
    return obj

@bots_routes.route("/update_collection", methods=["PUT"])
@cross_origin(origin='*')
@user_token_required
def update_collection(current_user):
    payload = request.json
    if payload["collection_content"] is None or payload["collection_name"] is None:
        return make_response(
            jsonify({"error": "Collection items must not be none"}), 400
        )
    file_path = "Documents/" + current_user["id"]
    file_name = payload["collection_name"] + ".txt"
    try:
        upload_to_blob_storage(file_path, file_name, payload["collection_name"] )
    except Exception as e:
        print(e)
        return make_response(jsonify({"error": "Cannot update the collection"}), 400)
    return make_response(jsonify({"data": "Collection updated"}), 200)


@bots_routes.route("/get_collections", methods=["GET"])
@cross_origin(origin='*')
@user_token_required
def get_collections(current_user):
    if 'my_files' not in current_user:
        return make_response(jsonify({"data": []}), 200)
    find_collection = [item['name'].split('.')[0] for item in current_user['my_files']]
    return make_response(jsonify({"data": find_collection}), 200)


@bots_routes.route("/get_collection", methods=["GET"])
@cross_origin(origin='*')
@user_token_required
def get_collection(current_user):
    args = request.args
    if args["collection_name"] is None:
        return make_response(jsonify({"error": "Must provide collection name"}), 400)
    if 'my_files' not in current_user:
        return make_response(jsonify({"error": "User doens't own this collection"}), 400)
    find_collection = next((item for item in current_user['my_files'] if item["name"] == args["collection_name"] + '.txt'), None)
    if find_collection is None:
        return make_response(jsonify({"error": "User doens't own this collection"}), 400)
    
    file_path = "Documents/" + current_user["id"]
    file_name = args["collection_name"] + ".txt"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(azure_container_name)
    blob_client = container_client.get_blob_client(file_path + '/' + file_name)
    res = blob_client.download_blob().readall()
    return make_response(jsonify({"data": str(res)}), 200)


@bots_routes.route("/delete_collection", methods=["DELETE"])
@cross_origin(origin='*')
@user_token_required
def delete_collection(current_user):
    args = request.args
    if args["collection_name"] is None:
        return make_response(jsonify({"error": "Must provide collection name"}), 400)
    file_path = "Documents/" + current_user["id"]
    file_name = args["collection_name"] + ".txt"
    if 'my_files' not in current_user:
        return make_response(jsonify({"error": "User doens't own this collection"}), 400)
    find_collection = next((item for item in current_user['my_files'] if item["name"] == file_name), None)
    if find_collection is None:
        return make_response(jsonify({"error": "User doens't own this collection"}), 400)

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
    
    current_user["my_files"] = [x for x in current_user["my_files"] if not (file_name == x['name'])]
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
    
    if 'my_files' not in current_user:
        return make_response(jsonify({"error": "User doens't own this collection"}), 400)
    find_collection = next((item for item in current_user['my_files'] if item["name"] == payload['collection_name'] + '.txt'), None)
    if find_collection is None:
        return make_response(jsonify({"error": "User doens't own this collection"}), 400)
    file_path = "Documents/" + current_user["id"]
    file_name = payload["collection_name"] + ".txt"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(azure_container_name)
    blob_client = container_client.get_blob_client(file_path + '/' + file_name)
    res = blob_client.download_blob().readall()
    new_node = []
    for item in res.splitlines():
        new_node.append({ "text": item})
    tmp_path = os.path.join(Path(__file__).parent.parent, 'utils/tmp/' + current_user["id"] + '_' + file_name)
    if not os.path.isdir(os.path.join(Path(__file__).parent.parent, 'utils/tmp')):
        os.mkdir(os.path.join(Path(__file__).parent.parent, 'utils/tmp'))
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    with open(tmp_path, 'a+') as output:
        output.write(json.dumps(new_node, indent=2, default=str, ensure_ascii=False))
    thread = Process(target=train_mode, args=(tmp_path, current_user, payload["collection_name"]))
    thread.start()
    return make_response(jsonify({"data": "Model training request submitted"}), 200)
    

@bots_routes.route("/add_collection", methods=["POST"])
@cross_origin(origin='*')
@user_token_required
def add_collection(current_user):
    print(request.headers['Content-Type'])
    payload = {}
    if "multipart/form-data" in request.headers['Content-Type']:
        payload["collection_content"] = "file"
        payload["collection_name"] = request.form.get("collection_name")
        payload["collection_type"] = request.form.get("collection_type")
    else:
        payload = request.json
    if payload["collection_content"] is None:
        return make_response(
            jsonify({"error": "Collection items must not be none"}), 400
        )
    json_file_name = payload["collection_name"] + ".txt"
    json_file_path = "Documents/" + current_user["id"]
    if 'my_files' in current_user:
        find_collection = next((item for item in current_user['my_files'] if item["name"] == json_file_name), None)
        if find_collection is not None:
            return make_response(
                jsonify({"error": "Collection name must be unique"}), 400
            )

    try:
        if payload['collection_type'] == 'link':
            with urllib.request.urlopen(payload["collection_content"]) as f:
                upload_to_blob_storage(json_file_path, json_file_name, f.read())
        elif payload['collection_type'] == 'file':
            file = request.files["collection_content"]
            upload_to_blob_storage(json_file_path, json_file_name, file.read())
        else:
            upload_to_blob_storage(json_file_path, json_file_name, payload["collection_content"])
    except Exception as e:
        print(e)
        return make_response(jsonify({"error": "Cannot save the collection"}), 400)
    db = get_client()
    users = db["users"]
    if "my_files" in current_user:
        current_user["my_files"].append({'name': json_file_name, 'model': ''})
    else:
        current_user["my_files"] = [{'name': json_file_name, 'model': ''}]
    users.update_one(
        {"id": current_user["id"]}, {"$set": {"my_files": current_user["my_files"]}}
    )
    return make_response(jsonify({"data": "New collection added"}), 201)


@bots_routes.route("/reset_context", methods=["POST"])
@cross_origin(origin='*')
@bot_api_key_required
def reset_context(current_user):
    reset_context_helper(current_user)
    return make_response(jsonify({"data": "Context refreshed"}), 200)


@bots_routes.route("/chat", methods=["POST"])
@cross_origin(origin='*')
@bot_api_key_required
def chat(current_user):
    payload = request.json
    if payload is None or payload["input"] is None:
        return make_response(jsonify({"error": "Must provide input key"}), 400)
    file_name = payload["collection_name"] + ".txt"
    #find_collection = next((item for item in current_user['my_files'] if item["name"] == file_name), None)
    #if find_collection is None:
      #  return make_response(jsonify({"error": "User does't have this file"}), 400)
    relevance_score = payload['relevance_score'] or 0.8
    file_name = "Documents/" + current_user["id"] + '/' + file_name
    result = pool.submit(
        asyncio.run, talk_bot(payload["input"], file_name, current_user, relevance_score * 100)
    ).result()
    #result = await talk_bot(payload["input"], file_name, payload["relevance_score"])
    return make_response(jsonify({"data": result}), 200)

@bots_routes.route("/get_google_calendars", methods=["POST"])
@cross_origin(origin='*')
@bot_api_key_required
def get_google_calendars(current_user):
    payload = request.json
    from src.main.routes.calendar_reader import GoogleCalendarReader
    from datetime import date

    loader = GoogleCalendarReader()
    documents = loader.load_data(start_date=date.today(), number_of_results=50)
    from typing import List
    from langchain.docstore.document import Document as LCDocument

    formatted_documents: List[LCDocument] = [doc.to_langchain_format() for doc in documents]
    #from langchain import OpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.memory import ConversationBufferMemory

    '''
    OpenAIEmbeddings uses text-embedding-ada-002
    '''

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(formatted_documents)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(documents, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    from langchain.chat_models import ChatOpenAI
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"), vector_store.as_retriever(), memory=memory)
    result = qa({"question": payload['input']})
    return make_response(jsonify({"data": result["answer"]}), 200)

'''@bots_routes.route("/tutor_agent", methods=["POST"])
@cross_origin(origin='*')
@bot_api_key_required
def tutor_agent(current_user):
    payload = request.json
    from langchain.prompts import load_prompt
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationChain

    prompt = load_prompt(os.path.join(os.getcwd(), 'src/main/routes/ranedeer.json'))
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', openai_api_key=api_key, openai_organization=org_id)
    conversation = ConversationChain(
            llm=llm, 
            prompt=prompt,
            verbose=True, 
            memory=ConversationBufferMemory(memory_key="tutor_history", return_messages=True),
        )
    response = conversation.predict(input=payload['input'])

    return make_response(jsonify({"data": response}), 200)'''
