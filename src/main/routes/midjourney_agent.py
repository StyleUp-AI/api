import os
import traceback
import time
import re
import requests
import json
import pandas as pd
from datetime import datetime
from azure.communication.email import EmailClient
import glob
from src.main.routes import discord_token, application_id, guild_id, channel_id, session_id, midjourney_waiting, get_client

class MidjourneyAgent:
    def __init__(self, current_user, prompt):
        self.prompt = prompt
        self.current_user = current_user
        self.user_id = current_user['id']
        self.channelid = channel_id
        self.authorization = discord_token
        self.application_id = application_id
        self.guild_id = guild_id
        self.session_id = session_id
        self.version = '1118961510123847772'
        self.id = '938956540159881230'
        self.waiting_images = []
        self.finished_images = []

    def send(self):
        print("Sending request to Discord")
        header = {
            'authorization': self.authorization
        }

        prompt = self.prompt.replace('_', ' ')
        prompt = " ".join(prompt.split())
        prompt = re.sub(r'[^a-zA-Z0-9\s]+', '', prompt)
        prompt = prompt.lower()

        payload = {'type': 2,
        'application_id': self.application_id,
        'guild_id': self.guild_id,
        'channel_id': self.channelid,
        'session_id': self.session_id,
        'data': {
            'version': self.version,
            'id': self.id,
            'name': 'imagine',
            'type': 1,
            'options': [{'type': 1, 'name': 'prompt', 'value': str(prompt)}],
            'attachments': []}
            }

        r = requests.post('https://discord.com/api/v9/interactions', json = payload , headers = header)
        while r.status_code != 204:
            r = requests.post('https://discord.com/api/v9/interactions', json = payload , headers = header)
            print(str(r.content))
        print('prompt [{}] successfully sent!'.format(prompt))

    def retrieve_messages(self):
        print("Get request from Discord")

        self.headers = {'authorization' : self.authorization}
        r = requests.get(
            f'https://discord.com/api/v10/channels/{self.channelid}/messages?limit={100}', headers=self.headers)
        jsonn = json.loads(r.text)
        return jsonn


    def collecting_results(self):
        while True:
            time.sleep(10)
            self.waiting_images = []
            print("Retrieving mid journey images")
            message_list  = self.retrieve_messages()
            for message in message_list:
                if (message['author']['username'] == 'Midjourney Bot') and ('**' in message['content']):
                    if len(message['attachments']) > 0:
                        if (message['attachments'][0]['filename'][-4:] == '.png') or ('(Open on website for full quality)' in message['content']):
                            id = message['id']
                            prompt = message['content'].split('**')[1].split(' --')[0]
                            url = message['attachments'][0]['url']
                            filename = message['attachments'][0]['filename']
                            found = False
                            for index, item in enumerate(self.user_images):
                                if item['id'] == id:
                                    found = True
                                    break
                            if not found:
                                self.finished_images.append({
                                    'id': id,
                                    'prompt': prompt,
                                    'url': url,
                                    'filename': filename,
                                    'status': 'ready',
                                    'is_downloaded': False
                                })
                        else:
                            id = message['id']
                            prompt = message['content'].split('**')[1].split(' --')[0]
                            if ('(fast)' in message['content']) or ('(relaxed)' in message['content']):
                                try:
                                    status = re.findall("(\w*%)", message['content'])[0]
                                except:
                                    status = 'unknown status'
                            self.waiting_images.append({
                                    'id': id,
                                    'prompt': prompt,
                                    'url': '',
                                    'filename': '',
                                    'status': status,
                                    'is_downloaded': False
                                })

                    else:
                        id = message['id']
                        prompt = message['content'].split('**')[1].split(' --')[0]
                        if '(Waiting to start)' in message['content']:
                            status = 'Waiting to start'
                        self.waiting_images.append({
                                'id': id,
                                'prompt': prompt,
                                'url': '',
                                'filename': '',
                                'status': status,
                                'is_downloaded': False
                            })
            if len(self.waiting_images) == 0:
                break

    def downloading_results(self):
        print("Downloading images")
        res = []
        for index, item in enumerate(self.finished_images):
            if item['status'] == 'ready' and item['is_downloaded'] == 0:
                res.append({
                'id': item['id'],
                'image_url': item['url'],
                'image_prompt': item['prompt']
                })
        db = get_client()
        users = db["users"]
        mid_images = db["mid_images"]

        if mid_images.count_documents({'user_id': self.current_user['id']}) > 0:
            old_images = mid_images.find_one({'user_id': self.current_user['id']})
            old_images['mid_images'].append(res)
            mid_images.update_one({'user_id': self.current_user['id']}, {'$set': {'mid_images': old_images['mid_images']}})
        else:
            mid_images.insert_one(
                {"user_id": self.current_user['id'], "mid_images": res}
            )
        print("mid journey images generated")
        azure_email_connection_string = os.environ.get("AZURE_EMAIL_CONNECTION_STRING")
        client = EmailClient.from_connection_string(azure_email_connection_string)
        message = {
            'content': {
                'subject': 'Your midjourney image is generated successfully!',
                'plainText': 'Hi, your midjourney image is generated successfully!',
                'html': 'Hi, your midjourney image is generated successfully!',
            },
            'recipients': {
                'to': [
                    {
                        'address': self.current_user["email"],
                        'displayName': 'Styleup AI'
                    }
                ]
            },
            'senderAddress': 'noreply@styleup.fun'
        }
        poller = client.begin_send(message)
        print(poller.result())

    def main(self):
        global midjourney_waiting
        while midjourney_waiting:
            print("Waiting in the queue")
            time.sleep(10)
        midjourney_waiting = True
        db = get_client()
        users = db["users"]
        mid_images = db["mid_images"]
        self.user_images = mid_images.find({'user_id': self.current_user['id']})
        print("Start midjourney image generation work")
        self.send()
        self.collecting_results()
        self.downloading_results()
        midjourney_waiting = False
