import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm  # Import timm for loading the EfficientNet model
from lightning.fabric.wrappers import _FabricModule  # Import the wrapper if it's part of Lightning

# Load the entire checkpoint, assuming it was saved with Lightning or another framework
checkpoint = torch.load('mobilenetv3_large_100_checkpoint_fold2.pt', map_location=torch.device('cpu'))

# Check if the checkpoint is wrapped in a Lightning Fabric module
if isinstance(checkpoint, _FabricModule):
    checkpoint = checkpoint.module.state_dict()

# Load the model structure
model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=4)
model.load_state_dict(checkpoint)
model.eval()
# Define the classes
classes = ['Fish', 'Flower', 'Gravel', 'Sugar']

# Define the transformation for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Title of the app
st.title("🎈 My new app")
# Description
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    # Transform the image
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    # Predict the class
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = classes[predicted.item()]
    # Display the prediction
    st.write(f"Prediction: {prediction}")