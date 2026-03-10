import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModel

MODEL_TAG = "conservationxlabs/miewid-msv2"

model = AutoModel.from_pretrained(MODEL_TAG, trust_remote_code=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((440, 440)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_vector(path):
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    print("TYPE:", type(output))
    try:
        print("SHAPE:", output.shape)
    except:
        print("NO DIRECT SHAPE")

    if isinstance(output, torch.Tensor):
        return output.cpu().numpy().flatten()

    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.cpu().numpy().flatten()

    if isinstance(output, (list, tuple)) and len(output) > 0:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first.cpu().numpy().flatten()

    raise Exception("Could not extract vector from model output")

def cosine(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

image_a = r"C:\Users\Liamh\Downloads\BCC_LF0002_right.jpg"
image_b = r"C:\Users\Liamh\Downloads\BCC_LF0002_right.jpg"
image_c = r"C:\Users\Liamh\OneDrive\Desktop\sharron\highlights\TQ2A2126-CR2_DxO_DeepPRIME-Edit.jpg"

vec_a = get_vector(image_a)
vec_b = get_vector(image_b)
vec_c = get_vector(image_c)

print("A vs B:", cosine(vec_a, vec_b))
print("A vs C:", cosine(vec_a, vec_c))