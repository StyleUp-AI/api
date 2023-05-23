from src.main.config import app_instance

app = app_instance

@app.route("/")
def home_route():
    return "Welcome to poke chat"
