import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from models.transformer_net import TransformerNet
from style_transfer.utils import load_image
import os

CONTENT_WEIGHT = 1e5
STYLE_WEIGHT = 1e10
EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
IMAGE_SIZE = 256
STYLE_IMAGE = 'style.jpg'
DATASET = 'data/train2014'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

train_dataset = datasets.ImageFolder(DATASET, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

vgg = models.vgg16(pretrained=True).features.to(device).eval()
content_layers = ['21']
style_layers = ['0', '5', '10', '19', '28']

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def get_features(x, model, layers):
    features = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

style_img = load_image(STYLE_IMAGE, size=IMAGE_SIZE).to(device)
style_features = get_features(style_img, vgg, style_layers)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

transformer = TransformerNet().to(device)
optimizer = optim.Adam(transformer.parameters(), LEARNING_RATE)
mse_loss = nn.MSELoss()

for epoch in range(EPOCHS):
    transformer.train()
    for batch_id, (x, _) in enumerate(train_loader):
        n_batch = len(x)
        x = x.to(device)
        y = transformer(x)

        y_features = get_features(y, vgg, content_layers + style_layers)
        x_features = get_features(x, vgg, content_layers)

        content_loss = CONTENT_WEIGHT * mse_loss(y_features[content_layers[0]], x_features[content_layers[0]])
        style_loss = 0
        for layer in style_layers:
            y_gram = gram_matrix(y_features[layer])
            style_gram = style_grams[layer].repeat(n_batch, 1, 1)
            style_loss += mse_loss(y_gram, style_gram)
        style_loss *= STYLE_WEIGHT

        total_loss = content_loss + style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (batch_id + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_id+1}], Loss: {total_loss.item():.4f}")

torch.save(transformer.state_dict(), "saved_model.pth")
print("Training complete. Model saved.")
