import torch
from torchvision import transforms
from PIL import Image

def load_image(img_path, size=None):
    image = Image.open(img_path).convert('RGB')
    if size:
        image = image.resize((size, size), Image.LANCZOS)
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)
    return image

def save_image(tensor, path):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)
