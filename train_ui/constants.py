LOCAL_SAVE_PATH = './.cache'

MODELS = [
    'CompVis/stable-diffusion-v1-4',
    'hakurei/waifu-diffusion',
    'runwayml/stable-diffusion-v1-5',
    'prompthero/openjourney',
    'Linaqruf/anything-v3.0',
    'stabilityai/stable-diffusion-2',
    'nitrosocke/mo-di-diffusion',
    'hassanblend/hassanblend1.4',
    'ogkalu/Comic-Diffusion'
    ]

MODEL_DICT = {
    'CompVis/stable-diffusion-v1-4' : 'Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.',
    'hakurei/waifu-diffusion' : 'waifu-diffusion is a latent text-to-image diffusion model that has been conditioned on high-quality anime images through fine-tuning.',
    'runwayml/stable-diffusion-v1-5' : 'The Stable-Diffusion-v1-5 checkpoint was initialized with the weights of the Stable-Diffusion-v1-2 checkpoint and subsequently fine-tuned on 595k steps at resolution 512x512 on "laion-aesthetics v2 5+" and 10% dropping of the text-conditioning to improve classifier-free guidance sampling.',
    'prompthero/openjourney' : 'Use prompt: mdjrny-v4 style',
    'Linaqruf/anything-v3.0' : 'This model is intended to produce high-quality, highly detailed anime style with just a few prompts. It also supports danbooru tags to generate images',
    'stabilityai/stable-diffusion-2' : 'This stable-diffusion-2 model is resumed from stable-diffusion-2-base (512-base-ema.ckpt) and trained for 150k steps using a v-objective on the same dataset.',
    'nitrosocke/mo-di-diffusion' : 'This is the fine-tuned Stable Diffusion model trained on high resolution 3D artworks. Use the tokens redshift style in your prompts for the effect.',
    'hassanblend/hassanblend1.4' : 'This model is intended to create photorealistic images of people.',
    'ogkalu/Comic-Diffusion' : 'This was created so anyone could create their comic projects with ease and flexibility. The tokens are:\n- charliebo artstyle\n- holliemengert artstyle\n- marioalberti artstyle\n- pepelarraz artstyle\n- andreasrocha artstyle\n- jamesdaly artstyle'
}