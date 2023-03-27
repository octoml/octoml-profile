# This example requires a torch nightly version `2.1.0.dev20230307`.
# It further requires `pip install datasets soundfile librosa`
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from octoml_profile import remote_profile, accelerate

#
# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# load dummy dataset and read audio files
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]


@accelerate(dynamic=True)
def predict(sample):
    input_features = processor(sample["array"],
                               sampling_rate=sample["sampling_rate"],
                               return_tensors="pt").input_features
    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription


with remote_profile(backends=["g4dn.xlarge/onnxrt-cuda", "r6i.large/onnxrt-cpu"]):
    for _ in range(3):
        text = predict(sample)
        print(text)
