from Long_CLIP.model import longclip
import torch


# At the module level
text_features = None

def get_features(text_prompts):
    global text_features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = longclip.load("Long_CLIP/checkpoints/longclip-B.pt", device=device)

    # Replace hardcoded prompts with the ones from config
    # Using the first two prompts from args.text_prompt (or fewer if only one exists)
    sample_prompts = text_prompts.text_prompt
    text = longclip.tokenize(sample_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = torch.stack([text_features, text_features])
    print("Text Features generated from Long_Clip") if len(text_features)!=0 else print("ERROR NO FEATURES FROM LONG_CLIP")