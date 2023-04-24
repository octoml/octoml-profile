from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from octoml_profile import accelerate, remote_profile

model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
backends = ['g4dn.xlarge/torch-eager-cuda[fp16]',
            'g5.xlarge/torch-eager-cuda[fp16]']


@accelerate
def predict(prompt):
    steps = 10
    images = pipe(prompt, num_inference_steps=steps).images
    return images


with remote_profile(backends=backends, num_repeats=1):
    for i in range(2):
        prompt = "A photo of an astronaut riding a horse on mars."
        predict(prompt)
