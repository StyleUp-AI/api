import os
from flask import Flask
from flask_cors import CORS
from src.main.routes import users, bots

app_instance = Flask(__name__)
cors = CORS(app_instance)
app_instance.register_blueprint(users.user_routes, url_prefix="/api/users")
app_instance.register_blueprint(bots.bots_routes, url_prefix="/api/bots")
