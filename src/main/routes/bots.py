import asyncio
import os
import json
import re
import asyncio, concurrent.futures
import speech_recognition as sr
from multiprocessing import Process
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage
from azure.storage.blob import BlobServiceClient
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoModel
from torch.nn import functional as F
from pathlib import Path
import pandas as pd
from PyPDF2 import PdfReader
from flask import Blueprint, request, jsonify, make_response
from flask_cors import cross_origin
from sentence_transformers import SentenceTransformer
from src.main.routes import user_token_required, bot_api_key_required, get_client, sk_prompt, connection_string, azure_container_name, user_sessions, upload_to_blob_storage
from src.main.utils.model_actions import train_mode
from src.main.routes.crawler import Crawler
from src.main.routes.midjourney_agent import MidjourneyAgent
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
from google_auth_oauthlib.flow import Flow

flow = Flow.from_client_secrets_file(
    os.path.join(os.getcwd(), "src/main/routes/credentials.json") , SCOPES, redirect_uri=os.environ.get("GOOGLE_REDIRECT_URL")
)
bots_routes = Blueprint("bots_routes", __name__)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
pool = concurrent.futures.ThreadPoolExecutor()

# Prepare OpenAI service using credentials stored in the `.env` file
api_key = os.environ.get("OPENAI_API_KEY")
org_id = os.environ.get("OPENAI_ORG_ID")

CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

def reset_context_helper(current_user):
    global user_sessions
    global sk_prompt
    user_sessions[current_user['id']] = {
        'context': {},
        'prompt_template': sk_prompt,
        'calendar_context': ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        'tutor_context': ConversationBufferMemory(return_messages=True),
        'audio_context': ConversationBufferMemory(return_messages=True),
        'blenderbot_context': []
    }

def reset_one_context(current_user, context):
    global user_sessions
    global sk_prompt
    if current_user['id'] not in user_sessions:
        reset_context_helper(current_user)
        return

    tmp_sessions = user_sessions
    if context == 'context':
        tmp_sessions[current_user['id']]['context'] = {}
    elif context == 'prompt_template':
        tmp_sessions[current_user['id']]['prompt_template'] = sk_prompt
    elif context == 'calendar_context':
        tmp_sessions[current_user['id']]['calendar_context'].clear()
    elif context == 'tutor_context':
        tmp_sessions[current_user['id']]['tutor_context'].clear()
        prompt = json.load(open(os.path.join(os.getcwd(), 'src/main/routes/ranedeer.json'), 'rb'))
        tmp_sessions[current_user['id']]['tutor_context'].chat_memory.add_ai_message(json.dumps(prompt, indent=2, default=str, ensure_ascii=False))
    elif context == 'audio_context':
        tmp_sessions[current_user['id']]['audio_context'].clear()
    elif context == 'blenderbot_context':
        tmp_sessions[current_user['id']]['blenderbot_context'] = []
    user_sessions = tmp_sessions

async def talk_bot(user_input, file_name, current_user, collection_name, relevance_score):
    global user_sessions
    global llm
    if current_user['id'] not in user_sessions:
        reset_context_helper(current_user)
    if file_name not in user_sessions[current_user['id']]['context']:
        user_sessions[current_user['id']]['context'][collection_name] = ConversationBufferMemory(return_messages=True)
    conversation_memory = user_sessions[current_user['id']]['context'][collection_name]
    az_loaders = AzureBlobStorageFileLoader(connection_string, azure_container_name, file_name)
    loaders = az_loaders.load()
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_rerank", return_intermediate_steps=True)
    results = chain({"input_documents": loaders, "question": user_input}, return_only_outputs=True)
    results = results["intermediate_steps"]
    max_score_item = max(results, key=lambda x:float(x['score']))
    if float(max_score_item['score']) >= relevance_score:
        conversation_memory.chat_memory.add_user_message(user_input)
        conversation_memory.chat_memory.add_ai_message(max_score_item['answer'])
        return max_score_item['answer']
    conversation = ConversationChain(memory=conversation_memory, prompt=user_sessions[current_user['id']]['prompt_template'], llm=OpenAI(temperature=0))
    obj = conversation.predict(input=user_input).split("AI: ",1)[1]
    conversation_memory.chat_memory.add_user_message(user_input)
    conversation_memory.chat_memory.add_ai_message(obj)
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
        upload_to_blob_storage(file_path, file_name, payload["collection_content"])
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

@bots_routes.route("/get_chat_history", methods=["GET"])
@cross_origin(origin='*')
@user_token_required
def get_chat_history(current_user):
    if current_user['id'] not in user_sessions:
        return make_response(jsonify({"data": []}), 200)
    args = request.args
    memory = None
    if args['chat_type'] != 'context':
        memory = user_sessions[current_user['id']][args['chat_type']]
    elif 'context' not in user_sessions[current_user['id']] or not user_sessions[current_user['id']]['context']:
        return make_response(jsonify({"data": []}), 200)
    else:
        if args['file_name'] in user_sessions[current_user['id']]['context']:
            memory = user_sessions[current_user['id']]['context'][args['file_name']]
        else:
            return make_response(jsonify({"data": []}), 200)
    prompt = json.load(open(os.path.join(os.getcwd(), 'src/main/routes/ranedeer.json'), 'rb'))
    data = memory.load_memory_variables({})
    res = []
    if 'history' in data:
        for item in data['history']:
            if type(item) is HumanMessage:
                res.append('Human: ' + item.content)
            elif item.content != json.dumps(prompt, indent=2, default=str, ensure_ascii=False):
                res.append('AIMessage: ' + item.content)
    elif 'chat_history' in data:
        for item in data['chat_history']:
            if type(item) is HumanMessage:
                res.append('Human: ' + item.content)
            elif item.content != json.dumps(prompt, indent=2, default=str, ensure_ascii=False):
                res.append('AIMessage: ' + item.content)
    return make_response(jsonify({"data": res}), 200)


@bots_routes.route("/get_collection", methods=["GET"])
@cross_origin(origin='*')
@user_token_required
def get_collection(current_user):
    args = request.args
    if args["collection_name"] is None:
        return make_response(jsonify({"error": "Must provide collection name"}), 400)
    if 'my_files' not in current_user:
        return make_response(jsonify({"error": "User doens't own this collection"}), 400)
    found = False
    for item in current_user['my_files']:
        if item['name'] == args["collection_name"] + '.txt':
            found = True
    if found == False:
        return make_response(jsonify({"error": "User doens't own this collection"}), 400)

    file_path = "Documents/" + current_user["id"]
    file_name = args["collection_name"] + ".txt"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(azure_container_name)
    blob_client = container_client.get_blob_client(file_path + '/' + file_name)
    res = blob_client.download_blob(max_concurrency=1, encoding='UTF-8').readall()
    return make_response(jsonify({"data": res}), 200)


@bots_routes.route("/delete_collection", methods=["DELETE"])
@cross_origin(origin='*')
@user_token_required
def delete_collection(current_user):
    args = request.args
    if args["collection_name"] is None:
        return make_response(jsonify({"error": "Must provide collection name"}), 400)
    file_path = "Documents/" + current_user["id"]
    print(current_user['my_files'])
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
            collection_name = json_file_name
            if "collection_name" in payload:
                collection_name = payload["collection_name"] + ".txt"
            crawler = Crawler(json_file_path, json_file_name, current_user, collection_name, [payload["collection_content"]], payload['link_levels'])
            thread = Process(target=crawler.run)
            thread.start()
            return make_response(jsonify({"data": "Web Crawler started"}), 200)
        elif payload['collection_type'] == 'file':
            file = request.files["collection_content"]
            file_name, file_extension = os.path.splitext(file.filename)
            if file_extension == '.pdf':
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                    text += '\n'
                upload_to_blob_storage(json_file_path, json_file_name, text)
            elif file_extension == '.xlsx':
                df_sheet_all = pd.read_excel(file, sheet_name=None)
                upload_to_blob_storage(json_file_path, json_file_name, str(df_sheet_all))
            else:
                content = file.read()
                upload_to_blob_storage(json_file_path, json_file_name, content)
        else:
            upload_to_blob_storage(json_file_path, json_file_name, payload["collection_content"])
    except Exception as e:
        print(e)
        return make_response(jsonify({"error": "Cannot save the collection"}), 400)
    db = get_client()
    users = db["users"]
    if "my_files" in current_user:
        current_user["my_files"].append({'name': payload["collection_name"] + ".txt", 'model': ''})
    else:
        current_user["my_files"] = [{'name': payload["collection_name"] + ".txt", 'model': ''}]
    users.update_one(
        {"id": current_user["id"]}, {"$set": {"my_files": current_user["my_files"]}}
    )
    return make_response(jsonify({"data": "New collection added"}), 201)


@bots_routes.route("/reset_context", methods=["POST"])
@cross_origin(origin='*')
@user_token_required
def reset_context(current_user):
    payload = request.json
    reset_one_context(current_user, payload["context"])
    return make_response(jsonify({"data": "Context refreshed"}), 200)


@bots_routes.route("/chat", methods=["POST"])
@cross_origin(origin='*')
@user_token_required
def chat(current_user):
    user_input = ""
    collection_name = ""
    relevance_score = 0.8

    if "multipart/form-data" in request.headers['Content-Type']:
        file = request.files["audio_file"]
        if "collection_name" in request.form:
            collection_name = request.form.get("collection_name")
        if "relevance_score" in request.form:
            relevance_score = request.form.get("relevance_score")
        try:
            r = sr.Recognizer()
            with sr.AudioFile(file) as source:
                audio_data = r.record(source)
                user_input = r.recognize_google(audio_data)
        except Exception as e:
            user_sessions[current_user['id']]['tutor_context'].chat_memory.add_ai_message(json.dumps("I can't understand, can you please repeat again?"))
            return make_response(jsonify({"data": "I can't understand, can you please repeat again?"}), 200)
    else:
        payload = request.json
        user_input = payload['input']
        collection_name = payload['collection_name']
        if 'relevance_score' in payload:
            relevance_score = payload['relevance_score']
    if user_input is None or user_input == "":
        return make_response(jsonify({"error": "Must provide input key"}), 400)
    file_name = collection_name + ".txt"
    #find_collection = next((item for item in current_user['my_files'] if item["name"] == file_name), None)
    #if find_collection is None:
      #  return make_response(jsonify({"error": "User does't have this file"}), 400)
    file_name = "Documents/" + current_user["id"] + '/' + file_name
    result = pool.submit(
        asyncio.run, talk_bot(user_input, file_name, current_user, collection_name, relevance_score * 100)
    ).result()
    #result = await talk_bot(payload["input"], file_name, payload["relevance_score"])
    return make_response(jsonify({"data": result}), 200)

@bots_routes.route("/authenticate_google_calendar", methods=["POST"])
@cross_origin(origins='*')
def authenticate_google_calendar():

    #creds = flow.run_local_server(port=8081)
    return make_response(jsonify({"data": flow.authorization_url(prompt='consent')}), 200)

@bots_routes.route("/authorize_session", methods=["GET"])
@cross_origin(origins='*')
def authorize_session():
    args = request.args
    code = args['code']
    flow.fetch_token(code=code)
    session = flow.credentials
    return make_response(jsonify({"data": session.to_json()}), 200)

@bots_routes.route("/get_google_calendars", methods=["POST"])
@cross_origin(origin='*')
@user_token_required
def get_google_calendars(current_user):
    user_input = ""
    user_info = ""
    if "multipart/form-data" in request.headers['Content-Type']:
        file = request.files["audio_file"]
        user_info = request.form.get("user_info")
        try:
            r = sr.Recognizer()
            with sr.AudioFile(file) as source:
                audio_data = r.record(source)
                user_input = r.recognize_google(audio_data)
        except Exception as e:
            user_sessions[current_user['id']]['tutor_context'].chat_memory.add_ai_message(json.dumps("I can't understand, can you please repeat again?"))
            return make_response(jsonify({"data": "I can't understand, can you please repeat again?"}), 200)
    else:
        payload = request.json
        user_input = payload['input']
        user_info = payload['user_info']
    from src.main.routes.calendar_reader import GoogleCalendarReader
    from datetime import date
    if current_user['id'] not in user_sessions:
        reset_context_helper(current_user)
    user_sessions[current_user['id']]['calendar_context'].chat_memory.add_user_message(user_input)
    loader = GoogleCalendarReader()
    if user_info == None or user_info == "":
        user_sessions[current_user['id']]['calendar_context'].chat_memory.add_ai_message("Need to login to google")
        return make_response(jsonify({"data": "Need to login to google"}), 200)

    documents = loader.load_data(user_info=json.loads(user_info))
    print(documents)
    session = user_sessions[current_user['id']]['calendar_context']
    from typing import List
    from langchain.docstore.document import Document as LCDocument

    formatted_documents: List[LCDocument] = [doc.to_langchain_format() for doc in documents]
    #from langchain import OpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.text_splitter import CharacterTextSplitter
    '''
    OpenAIEmbeddings uses text-embedding-ada-002
    '''

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(formatted_documents)

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(documents, embeddings)

    from langchain.chat_models import ChatOpenAI
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=2, model_name="gpt-3.5-turbo", openai_api_key=api_key, openai_organization=org_id), retriever=vector_store.as_retriever(), memory=session)
    result = qa({"question": user_input})
    user_sessions[current_user['id']]['calendar_context'].chat_memory.add_ai_message(result['answer'])
    return make_response(jsonify({"data": result["answer"]}), 200)

@bots_routes.route("/tutor_agent", methods=["POST"])
@cross_origin(origin='*')
@user_token_required
def tutor_agent(current_user):
    user_input = ""
    if "multipart/form-data" in request.headers['Content-Type']:
        file = request.files["audio_file"]
        try:
            r = sr.Recognizer()
            with sr.AudioFile(file) as source:
                audio_data = r.record(source)
                user_input = r.recognize_google(audio_data)
        except Exception as e:
            user_sessions[current_user['id']]['tutor_context'].chat_memory.add_ai_message(json.dumps("I can't understand, can you please repeat again?"))
            return make_response(jsonify({"data": "I can't understand, can you please repeat again?"}), 200)

    else:
        payload = request.json
        user_input = payload['input']
    prompt = json.load(open(os.path.join(os.getcwd(), 'src/main/routes/ranedeer.json'), 'rb'))
    if current_user['id'] not in user_sessions or 'tutor_context' not in user_sessions[current_user['id']]:
        reset_context_helper(current_user)
        user_sessions[current_user['id']]['tutor_context'].chat_memory.add_ai_message(json.dumps(prompt, indent=2, default=str, ensure_ascii=False))

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=api_key, openai_organization=org_id)
    conversation = ConversationChain(
            llm=llm,
            verbose=True,
            memory=user_sessions[current_user['id']]['tutor_context'],
        )
    response = conversation.predict(input=user_input)

    return make_response(jsonify({"data": response}), 200)

@bots_routes.route("/audio_agent", methods=["POST"])
@cross_origin(origin='*')
@user_token_required
def audio_agent(current_user):
    if current_user['id'] not in user_sessions:
        reset_context_helper(current_user)
    conversation_memory = user_sessions[current_user['id']]['audio_context']
    file = request.files["audio_file"]
    r = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', openai_api_key=api_key, openai_organization=org_id)
        conversation = ConversationChain(
                llm=llm,
                verbose=True,
                memory=user_sessions[current_user['id']]['audio_context'],
            )
        response = conversation.predict(input=text)

        conversation_memory.chat_memory.add_user_message(text)
        conversation_memory.chat_memory.add_ai_message(response)
        return make_response(jsonify({"data": response}), 200)

@bots_routes.route("/blenderbot_agent", methods=["POST"])
@cross_origin(origin='*')
@user_token_required
def blenderbot_agent(current_user):
    if current_user['id'] not in user_sessions:
        reset_context_helper(current_user)
    user_input = ""
    if "multipart/form-data" in request.headers['Content-Type']:
        file = request.files["audio_file"]
        try:
            r = sr.Recognizer()
            with sr.AudioFile(file) as source:
                audio_data = r.record(source)
                user_input = r.recognize_google(audio_data)
        except Exception as e:
            user_sessions[current_user['id']]['tutor_context'].chat_memory.add_ai_message(json.dumps("I can't understand, can you please repeat again?"))
            return make_response(jsonify({"data": "I can't understand, can you please repeat again?"}), 200)

    else:
        payload = request.json
        user_input = payload['input']
    conversation_memory = user_sessions[current_user['id']]['blenderbot_context']
    mname = "facebook/blenderbot-400M-distill"
    model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    utterance = user_input
    inputs = tokenizer(utterance, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    answer = tokenizer.decode(reply_ids[0])
    return make_response(jsonify({"data": cleanhtml(answer)}), 200)

@bots_routes.route("/midjourney_agent", methods=["POST"])
@cross_origin(origin='*')
@user_token_required
def midjourney_agent(current_user):
    payload = request.json
    prompt = payload['prompt']
    client = MidjourneyAgent(current_user, prompt)
    thread = Process(target=client.main)
    thread.start()
    return make_response(jsonify({"data": "Image is generating"}), 200)
