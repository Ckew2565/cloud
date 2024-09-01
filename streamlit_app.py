import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm  # ใช้ timm สำหรับโหลดโมเดล EfficientNet
from lightning.fabric.wrappers import _FabricModule  # นำเข้ารองรับหากเป็นส่วนหนึ่งของ Lightning

# โหลดเช็คพอยต์ทั้งหมด สมมติว่าได้บันทึกด้วย Lightning หรือเฟรมเวิร์กอื่น
checkpoint = torch.load('mobilenetv3_large_100_checkpoint_fold2.pt', map_location=torch.device('cpu'))

# ตรวจสอบว่าเช็คพอยต์ถูกห่อหุ้มในโมดูล Lightning Fabric หรือไม่
if isinstance(checkpoint, _FabricModule):
    checkpoint = checkpoint.module.state_dict()

# โหลดโครงสร้างโมเดล
model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=4)
model.load_state_dict(checkpoint)
model.eval()

# กำหนดชื่อคลาส
classes = ['Fish', 'Flower', 'Gravel', 'Sugar']

# กำหนดการแปลงภาพ
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ชื่อของแอป
st.title("☁️ Cloud Classification")

# คำอธิบาย
st.markdown("<h4 style='font-size: 24px;'>Please upload a satellite image.</h4>", unsafe_allow_html=True)

# อัปโหลดไฟล์สำหรับการป้อนภาพ
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # โหลดภาพ
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # แปลงภาพ
    image = transform(image)
    image = image.unsqueeze(0)  # เพิ่มมิติของแบทช์
    
    # ทำนายคลาส
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)  # ใช้ softmax เพื่อรับความน่าจะเป็น
        confidence, predicted = torch.max(probabilities, 1)
        prediction = classes[predicted.item()]
        confidence_percentage = confidence.item() * 100
    
    # แสดงการทำนาย
    st.write(f"Prediction Result is **{prediction}**")
    
    # แสดงความน่าจะเป็นสำหรับทุกคลาส
    for i, prob in enumerate(probabilities[0]):
        st.write(f"**{classes[i]}** : {prob.item() * 100:.2f}%")
