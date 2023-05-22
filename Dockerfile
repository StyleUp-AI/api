FROM --platform=linux/amd64 python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 3000
CMD ["flask", "--app", "app.py", "run", "-p","3000"]
