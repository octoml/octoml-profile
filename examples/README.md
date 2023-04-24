## Examples
For a slightly more complex example than the [simple tutorial example](../README.md#installation-and-getting-started),
we can take [the DistilBERT model from
HuggingFace](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english), where we make a
couple of lines of modification to the example code (`pip install transformers==4.27.4` is recommended).

To run other examples in this directory, please run (`pip install -r requirements.txt`) and pay
attention to the examples that require nightly torch. You can find recommended nightly 
torch version at [here](../README.md#dynamic-shapes).

```python
import torch
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

with remote_profile(backends=["r6i.large/onnxrt-cpu", "g5.xlarge/onnxrt-cuda"]):
    examples = [
      "Hello, world!",
      "Nice to meet you",
      "My dog is cute",
    ]
    for _ in range(3):
      for s in examples:
          predict(s)
```
And now we can easily run this model on a variety of hardware and understand
performance implications, all without having to worry about provisioning cloud
instances, configuring software or deploying our code.

You can use Dynamite directly within your application - whether it be a REST
API, CLI application or anything else - with your own data and tests.


### Dynamic models

We've enabled dynamic graph capture with `@accelerate(dynamic=True)`. See the
generative model [t5.py](t5.py), [gpt_neo_125m](gpt_neo_125m.py) and
[whisper](whisper.py) as examples.
