#!/usr/bin/env python
"""
Generate images using Stability AI API from WordNet-based prompts.
"""
import os
import json
import argparse
import requests
import time
from PIL import Image
from dotenv import load_dotenv

def load_prompts(prompt_file):
    """Load prompts from JSON file."""
    with open(prompt_file, 'r') as f:
        return json.load(f)

def send_generation_request(prompt, model="core", aspect_ratio="1:1", negative_prompt="", seed=0, output_format="png"):
    """Send a generation request to the Stability AI API."""
    host = f"https://api.stability.ai/v2beta/stable-image/generate/{model}"
    
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "aspect_ratio": aspect_ratio,
        "seed": seed,
        "output_format": output_format
    }
    
    # Send request
    print(f"Sending request to Stability AI {model.upper()} API...")
    response = requests.post(
        host,
        headers=headers,
        files={"none": ''},
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response

def generate_images(prompts, output_dir, model="core", aspect_ratio="1:1", 
                   num_images=1, seed=0, output_format="png", 
                   negative_prompt="", delay=3):
    """Generate images for each prompt using Stability AI API."""
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track progress
    total = len(prompts)
    success = 0
    failed = 0
    
    print(f"Generating images for {total} concepts using Stability AI {model.upper()} API...")
    
    for i, item in enumerate(prompts):
        prompt = item['prompt']
        name = item['name']
        
        # Create directory for this concept
        concept_dir = os.path.join(output_dir, item['path'] if 'path' in item and item['path'] else name)
        os.makedirs(concept_dir, exist_ok=True)
        
        print(f"[{i+1}/{total}] Generating {num_images} images for '{name}'...")
        
        # Generate requested number of images
        image_count = 0
        for img_idx in range(num_images):
            try:
                # Use a different seed for each image if seed is 0
                current_seed = seed if seed != 0 else int(time.time() * 1000) % 4294967295
                if img_idx > 0:
                    current_seed += img_idx
                
                # Make API request
                response = send_generation_request(
                    prompt=prompt,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    negative_prompt=negative_prompt,
                    seed=current_seed,
                    output_format=output_format
                )
                
                # Get response content
                output_image = response.content
                finish_reason = response.headers.get("finish-reason")
                actual_seed = response.headers.get("seed")
                
                # Check for NSFW classification
                if finish_reason == 'CONTENT_FILTERED':
                    print(f"  Image {img_idx+1} was filtered due to NSFW content")
                    continue
                
                # Save the image
                image_path = os.path.join(concept_dir, f"{name}_{img_idx:03d}_{actual_seed}.{output_format}")
                with open(image_path, "wb") as f:
                    f.write(output_image)
                
                print(f"  Saved image {img_idx+1}/{num_images} to {image_path}")
                image_count += 1
                
                # Add delay to avoid overloading the API
                if img_idx < num_images - 1 and delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"  Failed to generate image {img_idx+1} for '{name}': {e}")
        
        if image_count > 0:
            success += 1
        else:
            failed += 1
            
        # Add delay between concepts
        if i < total - 1 and delay > 0:
            time.sleep(delay)
    
    print(f"Completed: {success} successful, {failed} failed out of {total} concepts")
    return success, failed, total

if __name__ == "__main__":
    # Load environment variables
    load_dotenv(override=True)
    STABILITY_KEY = os.getenv("STABILITY_KEY")
    if not STABILITY_KEY:
        raise ValueError("STABILITY_KEY environment variable not found. Please create a .env file with your API key.")
    print("Using Stability API key:", STABILITY_KEY)
    
    parser = argparse.ArgumentParser(description="Generate images using Stability AI API")
    parser.add_argument("prompt_file", help="JSON file containing prompts")
    parser.add_argument("--output-dir", required=True, help="Output directory for images")
    parser.add_argument("--model", default="core", choices=["core", "ultra"], 
                        help="Stability AI model to use (core is faster, ultra is higher quality)")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate per concept")
    parser.add_argument("--aspect-ratio", default="1:1", 
                        choices=["21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"], 
                        help="Image aspect ratio")
    parser.add_argument("--seed", type=int, default=0, help="Seed for generation (0 for random)")
    parser.add_argument("--output-format", default="png", choices=["webp", "jpeg", "png"], help="Output format")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between API calls in seconds")
    
    args = parser.parse_args()
    
    # Load prompts
    prompts = load_prompts(args.prompt_file)
    
    # Generate images
    generate_images(
        prompts,
        args.output_dir,
        model=args.model,
        aspect_ratio=args.aspect_ratio,
        num_images=args.num_images,
        seed=args.seed,
        output_format=args.output_format,
        negative_prompt=args.negative_prompt,
        delay=args.delay
    )
