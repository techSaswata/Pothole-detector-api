from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model definition
class PotholeDetector(nn.Module):
    def __init__(self):
        super(PotholeDetector, self).__init__()
        # Load ResNet18 with pretrained weights
        weights = ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=weights)
        # Replace the last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)  # 2 classes: Plain and Pothole
    
    def forward(self, x):
        return self.resnet(x)

# Load trained model
model_path = "final_pothole_detector.pth"  # Update with your model file path
model = PotholeDetector().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded successfully!")

# Define data transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Pothole Detector API!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    try:
        # Load and preprocess the image
        image = Image.open(image_file.stream).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Perform prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
        
        # Map prediction to class label
        label = "Pothole" if predicted.item() == 1 else "Plain"
        return jsonify({"prediction": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
