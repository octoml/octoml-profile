from datasets import load_dataset
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from octoml_profile import accelerate, remote_profile

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
model_id = 'microsoft/resnet-50'
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = ResNetForImageClassification.from_pretrained(model_id)

inputs = feature_extractor(image, return_tensors="pt")


@accelerate
def run_model(inputs):
    return model(**inputs)


with remote_profile():
    for i in range(3):
        result = run_model(inputs)


predicted_label = result.logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
