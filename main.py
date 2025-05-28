from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

app = Flask(__name__)


# Define the model architecture
class NN(nn.Module):
  def __init__(self, input_feature):
    super().__init__()
    self.features = nn.Sequential(
        nn.Conv2d(input_feature, 32, kernel_size=3, padding='same'),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(32, 64, kernel_size=3, padding='same'),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 128, kernel_size=3, padding='same'),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32768, 128),
        nn.ReLU(),
        nn.Dropout(p=0.5),

        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(p=0.5),

        nn.Linear(64, 1),
        nn.Sigmoid()
    )



  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x

# Load model
model = NN(3)
model.load_state_dict(torch.load("cnn.pth", map_location=torch.device('cpu')))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files['file']
        if file:
            img = Image.open(file.stream).convert("RGB")
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(img)
                prob = output.item()
                prediction = f"Pneumonia Detected ({prob:.2f})" if prob > 0.5 else f"Normal ({1 - prob:.2f})"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
