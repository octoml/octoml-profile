from transformers import AutoTokenizer, BertForSequenceClassification
from octoml_profile import accelerate, remote_profile, RemoteInferenceSession

model_id = 'philschmid/MiniLM-L6-H384-uncased-sst2'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = BertForSequenceClassification.from_pretrained(model_id)

examples = [
    "Hello, world!",
    "Nice to meet you",
    "Goodbye, world!"
]
inputs = tokenizer(examples, return_tensors="pt")


model = accelerate(model)


session = RemoteInferenceSession(['r6i.large/onnxrt-cpu',
                                  'r6g.large/onnxrt-cpu'])
with remote_profile(session):
    for i in range(3):
        result = model(**inputs)

print(result.logits)
