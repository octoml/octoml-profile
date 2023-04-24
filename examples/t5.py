# This example requires torch nightly
# Recent nightly introduced a regression on onnx export:
#    https://github.com/pytorch/pytorch/issues/99788
# recommend nightly version torch<=2.1.0.dev20230327 for this example
# Please run `pip install -r requirements.txt`
from transformers import T5Tokenizer, T5ForConditionalGeneration
from octoml_profile import accelerate, remote_profile

model_id = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

input_text = "A step by step recipe to make bolognese pasta:"


@accelerate(dynamic=True)
def generate(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])


with remote_profile(backends=['g4dn.xlarge/onnxrt-cuda', 'r6i.large/onnxrt-cpu'],
                    num_repeats=1):
    for i in range(2):
        result = generate(input_text)
