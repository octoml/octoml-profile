from diffusers import StableDiffusionPipeline
from octoml_profile import accelerate, remote_profile

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
backends = ['g4dn.xlarge/torch-eager-cuda[fp16]',
            'g5.xlarge/torch-eager-cuda[fp16]']


pipe.unet = accelerate(pipe.unet)
pipe.vae.decode = accelerate(pipe.vae.decode)


def predict(prompt):
    steps = 10
    images = pipe(prompt, num_inference_steps=steps).images
    return images


with remote_profile(backends=backends, num_repeats=1):
    for i in range(2):
        prompt = "A photo of an astronaut riding a horse on marse."
        predict(prompt)
