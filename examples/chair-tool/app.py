"""Chair slider: a small design tool around one fine-tuned image model (FLUX.1 [dev] with a LoRA) and the
training-free slider from week 5. Run: python app.py, or from the week 6 notebook, which sets the settings below."""

import os, sys, io, tarfile, glob, hashlib
import gc, numpy as np, requests, torch
import gradio as gr
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from PIL import Image
dev = "cuda" if torch.cuda.is_available() else "mps"

# Settings. The week 6 notebook sets them for your model; without it, they are the week 5 museum chairs model.
LORA_URL = os.environ.get("LORA_URL", "https://replicate.delivery/xezq/KdQIrXFTvNZ0BdOrYZ1i1ovkskK4qiFV8ORnLlIpfX3RMuwLA/flux-lora.tar")
TRIGGER = os.environ.get("TRIGGER", "MUSCHR")
SUBJECT = os.environ.get("SUBJECT", "chair")
PROMPT = os.environ.get("PROMPT", f"a photo of a {TRIGGER} {SUBJECT}, museum object photograph on a plain grey backdrop")
PLUS = os.environ.get("PLUS", "ornate chair with carved decoration and tufted upholstery")
MINUS = os.environ.get("MINUS", "simple chair with straight lines and no decoration")

def get_lora(url):
    """Download the LoRA (a .safetensors file, or the .tar that Replicate's trainer returns) into its own folder."""
    folder = "lora/" + hashlib.sha1(url.encode()).hexdigest()[:8]
    found = glob.glob(folder + "/**/*.safetensors", recursive=True)
    if not found:
        os.makedirs(folder, exist_ok=True)
        data = requests.get(url, timeout=600).content
        if url.split("?")[0].endswith(".safetensors"):
            open(folder + "/lora.safetensors", "wb").write(data)
        else:
            tarfile.open(fileobj=io.BytesIO(data)).extractall(folder, filter="data")
        found = glob.glob(folder + "/**/*.safetensors", recursive=True)
    assert len(found) == 1, f"Expected one .safetensors file in {url}, found {len(found)}."
    return found[0]

lora_file = get_lora(LORA_URL)
print("LoRA:", lora_file)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", text_encoder=None, text_encoder_2=None,
                                    tokenizer=None, tokenizer_2=None, torch_dtype=torch.bfloat16).to(dev)
pipe.load_lora_weights(lora_file)

H, W, STEPS, START = 896, 704, 28, 7
_cache = {}

def encode(words):
    """Turn the prompt and the two slider ends into numbers for the image model. The text encoders are loaded only
    for this and freed right after, so the image model fits on the GPU. New words take about half a minute."""
    key = tuple(words)
    if key not in _cache:
        text = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", transformer=None, vae=None,
                                            torch_dtype=torch.bfloat16).to(dev)
        with torch.no_grad():
            _cache[key] = [text.encode_prompt(prompt=t, prompt_2=t, device=dev, max_sequence_length=256) for t in words]
        del text; gc.collect()
        torch.cuda.empty_cache() if dev == "cuda" else torch.mps.empty_cache()
    return _cache[key]

def make_image(plus, minus, s, seed=7):
    """One image with the slider at strength s (the target of Gandikota et al., ECCV 2024, Eq. 7, from step START on)."""
    base, more, less = encode([PROMPT, f"{PROMPT}, {plus}", f"{PROMPT}, {minus}"])
    g = torch.Generator("cpu").manual_seed(int(seed))
    lat, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, H, W, torch.bfloat16, dev, g, None)
    c = pipe.scheduler.config
    mu = calculate_shift(lat.shape[1], c.get("base_image_seq_len", 256), c.get("max_image_seq_len", 4096),
                         c.get("base_shift", 0.5), c.get("max_shift", 1.15))
    ts, _ = retrieve_timesteps(pipe.scheduler, STEPS, dev, sigmas=np.linspace(1.0, 1 / STEPS, STEPS), mu=mu)
    guidance = torch.full([1], 3.5, device=dev, dtype=torch.float32)
    def predict(cond, t):
        pe, pooled, ids = cond
        return pipe.transformer(hidden_states=lat, timestep=t.expand(1).to(lat.dtype) / 1000, guidance=guidance,
                                pooled_projections=pooled, encoder_hidden_states=pe, txt_ids=ids, img_ids=img_ids,
                                return_dict=False)[0]
    with torch.no_grad():
        for i, t in enumerate(ts):
            v = predict(base, t)
            if i >= START and s != 0:
                v = v + s * (predict(more, t) - predict(less, t))
            lat = pipe.scheduler.step(v, t, lat, return_dict=False)[0]
        lat = pipe._unpack_latents(lat, H, W, pipe.vae_scale_factor)
        lat = lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        return pipe.image_processor.postprocess(pipe.vae.decode(lat, return_dict=False)[0], output_type="pil")[0]

with gr.Blocks(title=f"{SUBJECT.capitalize()} slider") as demo:
    gr.Markdown(f"## {SUBJECT.capitalize()} slider")
    with gr.Row():
        with gr.Column():
            more = gr.Textbox(label="More of", value=PLUS)
            less = gr.Textbox(label="Less of", value=MINUS)
            strength = gr.Slider(-2, 2, value=0, step=0.25, label="Slider")
            seed = gr.Number(value=7, precision=0, label="Seed")
            go = gr.Button("Make image", variant="primary")
        out = gr.Image(label="Result", type="pil")
    go.click(make_image, inputs=[more, less, strength, seed], outputs=out)

if __name__ == "__main__":
    demo.launch()
