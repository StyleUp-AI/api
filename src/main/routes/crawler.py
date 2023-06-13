from urllib.parse import urljoin
import os
import requests
from azure.communication.email import EmailClient
from src.main.routes import upload_to_blob_storage, get_client
from bs4 import BeautifulSoup

class Crawler:
    def __init__(self, file_path, file_name, current_user, urls=[], layers=1):
        self.visited_urls = []
        self.urls_to_visit = urls
        self.layers = layers
        self.content = []
        self.file_path = file_path
        self.file_name = file_name
        self.current_user = current_user
    
    def download_url(self, url):
        return requests.get(url).text
    
    def get_linked_urls(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
         # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        self.content.append(soup.get_text())
        for link in soup.find_all('a'):
            path = link.get('href')
            if path and path.startswith('/'):
                path = urljoin(url, path)
            yield path
    
    def add_url_to_visit(self, url):
        if url not in self.visited_urls and url not in self.urls_to_visit:
            self.urls_to_visit.append(url)
    
    def crawl(self, url):
        html = self.download_url(url)
        for url in self.get_linked_urls(url, html):
            self.add_url_to_visit(url)

    def run(self):
        counter = 0
        self.content = []
        file_content = ''
        while self.urls_to_visit and counter < self.layers:
            url = self.urls_to_visit.pop(0)
            print(f'Crawling: {url}')
            try:
                self.crawl(url)
            except Exception:
                print(f'Failed to crawl: {url}')
            finally:
                self.visited_urls.append(url)
                counter += 1
        for item in self.content:
            file_content += item
            file_content += '\n'
        print(file_content)
        upload_to_blob_storage(self.file_path, self.file_name, file_content)
        db = get_client()
        users = db["users"]
        if "my_files" in self.current_user:
            self.current_user["my_files"].append({'name': self.file_name, 'model': ''})
        else:
            self.current_user["my_files"] = [{'name': self.file_name, 'model': ''}]
        users.update_one(
            {"id": self.current_user["id"]}, {"$set": {"my_files": self.current_user["my_files"]}}
        )
        azure_email_connection_string = os.environ.get("AZURE_EMAIL_CONNECTION_STRING")
        client = EmailClient.from_connection_string(azure_email_connection_string)
        message = {
            'content': {
                'subject': 'Your web connection was created successfully!',
                'plainText': 'Hi, Your web link collection was created successfully!',
                'html': 'Hi, Your web link collection was created successfully!',
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
        
        