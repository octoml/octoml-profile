## Examples

For a slightly more complex example than the [simple tutorial example](../README.md#getting-started),
we can take [the DistilBERT model from
HuggingFace](https://huggingface.co/distilbert-base-uncased), where we make a
couple of lines of modification to the example code (`pip install transformers==4.25.0`
is required):

```python
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
```
And now we can easily run this model on a variety of hardware and understand
performance implications, all without having to worry about provisioning cloud
instances, configuring software or deploying our code.

You can use Dynamite directly within your application - whether it be a REST
API, CLI application or anything else - with your own data and tests.


### Model coverage

We are actively working on enabling models like Stable Diffusion and GPT2.
See the [Known issues section](../README.md#known-issues) for a sense
of octoml-profile's model coverage surface. If you have a use case that we
don't currently support, please file an issue and we will prioritize support accordingly.
