from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from octoml_profile import accelerate, remote_profile


model_id = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_id)
model = DistilBertForSequenceClassification.from_pretrained(model_id)


@accelerate
def predict(input: str):
    inputs = tokenizer(input, return_tensors="pt")
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]


with remote_profile(backends=["r6i.large/torch-eager-cpu",
                              "g4dn.xlarge/torch-eager-cuda",
                              "g4dn.xlarge/onnxrt-cuda"]):
    examples = [
        "Hello, world!",
        "Nice to meet you",
        "My dog is cute",
    ]
    for _ in range(3):
        for s in examples:
            predict(s)
