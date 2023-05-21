FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 3000
CMD ["flask", "--app", "index.py", "run", "-p","3000"]
