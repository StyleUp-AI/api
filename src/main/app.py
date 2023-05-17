from flask import Flask
from src.main.config import app_instance

flask_app = app_instance

@flask_app.route("/")
def home_route():
    return "Welcome to poke chat"
