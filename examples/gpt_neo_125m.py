# This example requires torch nightly (see README.md for recommended version)

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from octoml_profile import accelerate, remote_profile

model_id = "EleutherAI/gpt-neo-125M"
model = GPTNeoForCausalLM.from_pretrained(model_id)
tokenizer = GPT2Tokenizer.from_pretrained(model_id)

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)


@accelerate(dynamic=True)
def predict(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    return tokenizer.batch_decode(gen_tokens)[0]


with remote_profile(backends=["g5.xlarge/onnxrt-cuda"], num_repeats=1):
    for i in range(3):
        predict(prompt)
