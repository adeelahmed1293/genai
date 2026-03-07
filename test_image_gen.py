"""
Simple test — checks if Stable Diffusion image generation works on your system.
Generates one image and saves it in the same folder as this script.

INSTALL (run once):
    pip install diffusers transformers accelerate torch torchvision Pillow

RUN:
    python3 test_image_gen.py
"""

import os
import time

def test_image_generation():
    print("=" * 50)
    print("  STABLE DIFFUSION — IMAGE GENERATION TEST")
    print("=" * 50)

    # Step 1 — Check dependencies
    print("\n[1/3] Checking dependencies...")
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        print(f"  ✅ torch      : {torch.__version__}")
        import diffusers
        print(f"  ✅ diffusers  : {diffusers.__version__}")
    except ImportError as e:
        print(f"  ❌ Missing package: {e}")
        print("\n  Fix:")
        print("  pip install diffusers transformers accelerate torch torchvision Pillow")
        return

    # Step 2 — Load model
    print("\n[2/3] Loading Stable Diffusion 2.1 model...")
    print("      First time: downloads ~5 GB to ~/.cache/huggingface/")
    print("      After that: loads from local cache in ~30s")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32
        print(f"      Device: {device.upper()}")

        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)

        if device == "cpu":
            pipe.enable_attention_slicing()  # reduces RAM usage on CPU

        print("  ✅ Model loaded successfully")
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return

    # Step 3 — Generate image
    print("\n[3/3] Generating image (steps=3 for quick test)...")
    if device == "cpu":
        print("      Running on CPU — this takes ~2-5 minutes, please wait...")
    else:
        print("      Running on GPU — this takes ~10-20 seconds...")

    try:
        t_start = time.time()
        image = pipe(
            "A photorealistic mountain landscape at golden hour, snow-capped peaks",
            num_inference_steps=3,   # 3 steps = fast test only
            width=512,
            height=512
        ).images[0]
        elapsed = time.time() - t_start

        # Save in same folder as this script
        save_dir  = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(save_dir, "test_generated_image.png")
        image.save(save_path)

        print(f"\n  ✅ Image generated in {elapsed:.1f}s")
        print(f"  ✅ Saved to: {save_path}")
        print("\n" + "=" * 50)
        print("  SUCCESS — Stable Diffusion is working!")
        print("  You can now run the full benchmark script.")
        print("=" * 50)

    except Exception as e:
        print(f"  ❌ Image generation failed: {e}")


if __name__ == "__main__":
    test_image_generation()
