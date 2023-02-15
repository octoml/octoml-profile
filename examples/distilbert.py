import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from octoml_profile import (accelerate,
                            remote_profile,
                            RemoteInferenceSession)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

@accelerate
def predict(input: str):
  inputs = tokenizer(input, return_tensors="pt")
  logits = model(**inputs).logits

  predicted_class_id = logits.argmax().item()
  return model.config.id2label[predicted_class_id]

session = RemoteInferenceSession(backends=["r6i.large/onnxrt-cpu", "g5.xlarge/onnxrt-cuda"])
with remote_profile(session):
    examples = [
      "Hello, world!",
      "Nice to meet you",
      "My dog is cute",
    ]
    for s in examples:
        predict(s)
