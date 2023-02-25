from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler
import torch
#Return a CheckpointMergerPipeline class that allows you to merge checkpoints. 
#The checkpoint passed here is ignored. But still pass one of the checkpoints you plan to 
#merge for convenience
pipe = DiffusionPipeline.from_pretrained("refined_checkpoint", custom_pipeline="checkpoint_merger")

#There are multiple possible scenarios:
#The pipeline with the merged checkpoints is returned in all the scenarios

#Compatible checkpoints a.k.a matched model_index.json files. Ignores the meta attributes in model_index.json during comparision.( attrs with _ as prefix )
merged_pipe = pipe.merge(["refined_checkpoint","realistic_vision_v1.3_checkpoint"], interp = "sigmoid", alpha = 0.4)
merged_pipe.save_pretrained("realistic_refined_merged")
merged_pipe.to("cuda")

#Incompatible checkpoints in model_index.json but merge might be possible. Use force = True to ignore model_index.json compatibility
# merged_pipe_1 = pipe.merge(["refined_checkpoint","hakurei/waifu-diffusion"], force = True, interp = "sigmoid", alpha = 0.4)

#Three checkpoint merging. Only "add_difference" method actually works on all three checkpoints. Using any other options will ignore the 3rd checkpoint.
# merged_pipe_2 = pipe.merge(["refined_checkpoint","hakurei/waifu-diffusion","prompthero/openjourney"], force = True, interp = "add_difference", alpha = 0.4)
generator = torch.Generator("cuda").manual_seed(47)
prompt = "RAW photo, a close up portrait photo of brutal 45 y.o man in wastelander clothes, long haircut, pale skin, slim body, background is city ruins, high detailed skin, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
negative_prompt = "nude, naked, deformed, bad hands, bad fingers, monochrome:1.3, oversaturated:1.3, bad hands, lowers, 3d render, cartoon, long body, blurry, duplicate, duplicate body parts, disfigured, poorly drawn, extra limbs, fused fingers, extra fingers, twisted, malformed hands, mutated hands and fingers, contorted, conjoined, missing limbs, logo, signature, text, words, low res, boring, mutated, artifacts, bad art, gross, ugly, poor quality, low quality, child"
merged_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(merged_pipe.scheduler.config)
image = merged_pipe(prompt=prompt,negative_prompt=negative_prompt,guidance_scale=7.0, num_inference_steps=25, generator=generator).images[0]
image.save("merged.png")
