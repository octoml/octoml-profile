from transformers import AutoTokenizer, BertForSequenceClassification
from octoml_profile import accelerate, remote_profile

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


with remote_profile(backends=["r6i.large/onnxrt-cpu",
                              "r6i.large/torch-eager-cpu",
                              "r7g.large/onnxrt-cpu",
                              "g4dn.xlarge/onnxrt-cuda",
                              "g4dn.xlarge/onnxrt-tensorrt",
                              "g4dn.xlarge/torch-eager-cuda",
                              "g4dn.xlarge/torch-inductor-cuda",
                              "g5.xlarge/torch-eager-cuda"]):
    for i in range(3):
        result = model(**inputs)

print(result.logits)
