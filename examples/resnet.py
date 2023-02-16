# This example requires the following packages to be installed
# `pip install datasets`

from datasets import load_dataset
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from octoml_profile import accelerate, remote_profile, RemoteInferenceSession

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
model_id = 'microsoft/resnet-50'
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = ResNetForImageClassification.from_pretrained(model_id)

inputs = feature_extractor(image, return_tensors="pt")


@accelerate
def run_model(inputs):
    return model(**inputs)


session = RemoteInferenceSession(['g4dn.xlarge/onnxrt-cuda',
                                  'g4dn.xlarge/onnxrt-tensorrt',
                                  'g5.xlarge/onnxrt-cuda',
                                  'g5.xlarge/onnxrt-tensorrt'])
with remote_profile(session):
    for i in range(3):
        result = run_model(inputs)


predicted_label = result.logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
