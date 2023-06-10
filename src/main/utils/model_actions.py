from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from azure.storage.blob import BlobServiceClient
from src.main.utils import connection_string, azure_container_name
from src.main.routes import get_client
from azure.communication.email import EmailClient
import os
import shutil
import numpy as np
import evaluate

def train_mode(path, current_user, collection_name):
    # download and prepare cc_news dataset
    #dataset = load_dataset("cc_news", split="train")
    model_path = os.path.join(os.getcwd(), 'src/main/utils/models')
    new_model_path = model_path + '/' + current_user['id'] + '_' + collection_name
    archive_path = os.path.join(os.getcwd(), 'src/main/utils/models/' + current_user['id'] + '_archive_' + collection_name)
    try:
        os.remove(archive_path + '.zip')
        shutil.rmtree(new_model_path)
        os.remove('test.txt')
        os.remove('train.txt')
    except OSError:
        pass
    dataset = load_dataset("json", data_files=path, split="train")
    d = dataset.train_test_split(test_size=0.1)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    def tokenize_function(input):
        return tokenizer(input['text'], padding='max_length', truncation=True)
    train_dataset = d['train'].map(tokenize_function, batched=True)
    test_dataset = d['test'].map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=20)
    training_args = TrainingArguments(output_dir=new_model_path, evaluation_strategy="epoch")
    metric = evaluate.load('accuracy')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(new_model_path)

    shutil.make_archive(archive_path, 'zip', new_model_path)
    azure_file_path = "Documents/" + current_user["id"] + '/models'
    azure_file_name = collection_name
    destination = azure_file_path + '/' + azure_file_name
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=azure_container_name, blob=destination)
    with open(archive_path + '.zip', 'rb') as data:
        blob_client.upload_blob(data, overwrite=True)
        print('Uploaded model: ' + collection_name)
    try:
        os.remove(path)
        os.remove(archive_path + '.zip')
        shutil.rmtree(new_model_path)
        os.remove('test.txt')
        os.remove('train.txt')
    except OSError:
        pass
    db = get_client()
    users = db["users"]
    for item in current_user["my_files"]:
        if item["name"] == collection_name:
            item['model'] = destination
    users.update_one({'id': current_user['id']}, {'$set': {'my_files': current_user['my_files']}})
    send_email(current_user['email'], collection_name, blob_client.url)
    print('Finished' + destination)
    

model_html_template = """\
<html>
  <body>
    <p>Hi,<br>
       Your model {0} has been generated successfully, here is the link: {1}
    </p>
  </body>
</html>
"""

def send_email(email, model_name, url):
    if email is None:
        print("Must provide email")
        return 
    azure_email_connection_string = os.environ.get("AZURE_EMAIL_CONNECTION_STRING")
    client = EmailClient.from_connection_string(azure_email_connection_string)
    message = {
        'content': {
            'subject': 'Your model ' + model_name + ' has been generated successfully',
            'plainText': 'Hi, Your model' + model_name + ' has been generated successfully, here is the link: ' + url,
            'html': model_html_template.format(model_name, url)
        },
        'recipients': {
            'to': [
                {
                    'address': email,
                    'displayName': 'Styleup AI'
                }
            ]
        },
        'senderAddress': 'noreply@styleup.fun'
    }
    poller = client.begin_send(message)
    print(poller.result())
       
    
'''
# when you load from pretrained
model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-10000"))
tokenizer = BertTokenizerFast.from_pretrained(model_path)
# or simply use pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# perform predictions
example = "Who is [MASK]"
for prediction in fill_mask(example):
  print(prediction)

# perform predictions
examples = [
  "Why Mengjiao Wang walking with [MASK]",
  "The [MASK] was cloudy yesterday, but today it's rainy.",
]
for example in examples:
  for prediction in fill_mask(example):
    print(f"{prediction['sequence']}, confidence: {prediction['score']}")
  print("="*50)
'''