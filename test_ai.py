import open_clip
import torch

print("Loading the AI model...")
print("This may take a few minutes the first time.")

try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model.eval()

    print("\nSUCCESS! The AI model loaded correctly.")
    print("Device:", "GPU" if torch.cuda.is_available() else "CPU")
    print("The Pokémon recognition system is ready for the next step.")

except Exception as error:
    print("\nThe AI model could not be loaded.")
    print("Error details:")
    print(error)