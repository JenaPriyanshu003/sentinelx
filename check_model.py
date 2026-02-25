import torch
import timm

model = timm.create_model("xception", pretrained=False, num_classes=2)

state = torch.load("models/xception_pretrained.pth", map_location="cpu")
model.load_state_dict(state, strict=False)

model.eval()

x = torch.randn(1, 3, 299, 299)
y = model(x)

print("Model loaded successfully. Output shape:", y.shape)