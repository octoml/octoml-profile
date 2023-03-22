from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from octoml_profile import accelerate, remote_profile


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")


@accelerate
def predict(input: str):
    inputs = tokenizer(input, return_tensors="pt")
    logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]


with remote_profile(backends=["r6i.large/torch-eager-cpu",
                              "g4dn.xlarge/torch-eager-cuda",
                              "g4dn.xlarge/torch-inductor-cuda",
                              "g4dn.xlarge/onnxrt-cuda"]):
    examples = [
        "Hello, world!",
        "Nice to meet you",
        "My dog is cute",
    ]
    for s in examples:
        predict(s)
