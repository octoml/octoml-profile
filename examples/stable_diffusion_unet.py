from diffusers import StableDiffusionPipeline
from octoml_profile import accelerate, remote_profile

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Only measure UNet of the pipeline
pipe.unet = accelerate(pipe.unet)


def predict(prompt):
    batch_size = 1
    steps = 10
    images = pipe(prompt, num_inference_steps=steps, num_images_per_prompt=batch_size).images
    return images


with remote_profile(backends=['g5.xlarge/torch-eager-cuda', 'g5.xlarge/torch-inductor-cuda']):
    for i in range(2):
        prompt = "A photo of an astronaut riding a horse on marse."
        predict(prompt)
