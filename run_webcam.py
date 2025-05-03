import cv2
import torch
from torchvision import transforms
from models.transformer_net import TransformerNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerNet()
model.load_state_dict(torch.load("saved_model.pth"))
model.to(DEVICE)
model.eval()

cap = cv2.VideoCapture(0)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(Image.fromarray(img)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor).cpu()
    output = output.squeeze(0).clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype('uint8')
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    cv2.imshow('Stylized Webcam', output_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
