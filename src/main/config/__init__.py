import os
from flask import Flask
from src.main.routes import users, bots

app_instance = Flask(__name__)
app_instance.register_blueprint(users.user_routes, url_prefix="/api/users")
app_instance.register_blueprint(bots.bots_routes, url_prefix="/api/bots")
app_instance.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")
