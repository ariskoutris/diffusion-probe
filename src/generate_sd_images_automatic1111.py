#!/usr/bin/env python
"""
Generate images using Automatic1111's Stable Diffusion API from WordNet-based prompts.
Set up here: https://github.com/AUTOMATIC1111/stable-diffusion-webui
"""
import os
import json
import argparse
import requests
import io
import base64
import time
from PIL import Image, PngImagePlugin

def load_prompts(prompt_file):
    """Load prompts from JSON file."""
    with open(prompt_file, 'r') as f:
        return json.load(f)

def generate_images(prompts, output_dir, api_url, 
                   batch_size=4, num_images=8, steps=35, cfg_scale=5.5, 
                   sampler="LMS", width=512, height=512, seed=-1, delay=1):
    """Generate images for each prompt using Stable Diffusion API."""
    txt2img_endpoint = f'{api_url}/sdapi/v1/txt2img'
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track progress
    total = len(prompts)
    success = 0
    failed = 0
    
    print(f"Generating images for {total} concepts...")
    
    for i, item in enumerate(prompts):
        prompt = item['prompt']
        name = item['name']
        
        # Create directory for this concept
        concept_dir = os.path.join(output_dir, item['path'] if item['path'] else name)
        os.makedirs(concept_dir, exist_ok=True)
        
        print(f"[{i+1}/{total}] Generating {num_images} images for '{name}'...")
        
        # Set up payload for API request
        payload = {
            "prompt": prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "batch_size": batch_size,
            "n_iter": (num_images + batch_size - 1) // batch_size,  # Ceiling division
            "width": width,
            "height": height,
            "sampler_index": sampler
        }
        
        try:
            # Make API request
            response = requests.post(url=txt2img_endpoint, json=payload)
            if response.status_code != 200:
                print(f"  Error: API returned status code {response.status_code}")
                failed += 1
                continue
                
            r = response.json()
            
            # Process and save images
            image_count = 0
            for img_data in r['images']:
                try:
                    image = Image.open(io.BytesIO(base64.b64decode(img_data.split(",",1)[0])))
                    
                    # Get image metadata
                    png_payload = {
                        "image": "data:image/png;base64," + img_data
                    }
                    response2 = requests.post(url=f'{api_url}/sdapi/v1/png-info', json=png_payload)
                    
                    # Save image with metadata
                    pnginfo = PngImagePlugin.PngInfo()
                    pnginfo.add_text("parameters", response2.json().get("info"))
                    
                    # Save the image
                    image_path = os.path.join(concept_dir, f"{name}_{image_count:03d}.png")
                    image.save(image_path, pnginfo=pnginfo)
                    image_count += 1
                except Exception as e:
                    print(f"  Error processing image: {e}")
            
            print(f"  Saved {image_count} images to {concept_dir}")
            success += 1
            
            # Add delay to avoid overloading the API
            if i < total - 1 and delay > 0:
                time.sleep(delay)
                
        except Exception as e:
            print(f"  Failed to generate images for '{name}': {e}")
            failed += 1
    
    print(f"Completed: {success} successful, {failed} failed out of {total} concepts")
    return success, failed, total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion API")
    parser.add_argument("prompt_file", help="JSON file containing prompts")
    parser.add_argument("--output-dir", required=True, help="Output directory for images")
    parser.add_argument("--api-url", default="http://localhost:7860", help="Stable Diffusion API URL")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--num-images", type=int, default=8, help="Number of images to generate per concept")
    parser.add_argument("--steps", type=int, default=35, help="Number of sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=5.5, help="CFG scale")
    parser.add_argument("--sampler", default="LMS", help="Sampler to use")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for generation (-1 for random)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls in seconds")
    
    args = parser.parse_args()
    
    # Load prompts
    prompts = load_prompts(args.prompt_file)
    
    # Generate images
    generate_images(
        prompts,
        args.output_dir,
        api_url=args.api_url,
        batch_size=args.batch_size,
        num_images=args.num_images,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        sampler=args.sampler,
        width=args.width,
        height=args.height,
        seed=args.seed,
        delay=args.delay
    )
