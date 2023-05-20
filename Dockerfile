FROM python:3.11 as base
WORKDIR /deploy/
COPY . /deploy/
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 3000
CMD ["flask", "--app", "index.py", "run", "-p","3000"]
