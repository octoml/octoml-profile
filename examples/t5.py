# This example requires the following packages to be installed
# `pip install sentencepiece`

# Currently, the remote execution will execute graph with static shape
# which means for each decoding step is treated as a new graph. We
# are actively working on support dynamic shape execution.

from transformers import T5Tokenizer, T5ForConditionalGeneration
from octoml_profile import accelerate, remote_profile, RemoteInferenceSession

model_id = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

input_text = "translate English to German: How old are you?"

@accelerate
def generate(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])


session = RemoteInferenceSession([
    'g5.xlarge/onnxrt-cuda',
    'g5.xlarge/onnxrt-tensorrt',
])
with remote_profile(session):
    for i in range(2):
        result = generate(input_text)
    print(result)
