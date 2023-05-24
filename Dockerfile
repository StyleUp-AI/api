FROM --platform=linux/amd64 python:3.8-slim-buster
WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt
EXPOSE 3000
CMD ["flask", "--app", "app.py", "run", "-p","3000"]
