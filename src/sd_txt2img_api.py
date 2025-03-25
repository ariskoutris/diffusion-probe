# %%
import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin

# %%
url = "http://localhost:7860"
txt2img_endpoint = f'{url}/sdapi/v1/txt2img'

# %% [markdown]
# Note: batch_size = 4 is faster than batch_size = 1. 
# To Do: Test bigger batch sizes.

# %%
payload = {
  "prompt": "a shiba inu",
  "steps": 50,
  "cfg_scale": 7,
  "seed": -1,
  "batch_size": 4,
  "n_iter": 1,
  "negative_prompt": "",
  "width": 512,
  "height": 512,
  "sampler_index": "Euler a",
  "restore_faces": False,
  "tiling": False,
  "enable_hr": False
}

# %%
response = requests.post(url=txt2img_endpoint, json=payload)
r = response.json()

# %%
for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)
    
    image.show()
    
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", response2.json().get("info"))
    image.save('output.png', pnginfo=pnginfo)


