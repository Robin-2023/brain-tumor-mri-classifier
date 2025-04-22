import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
import time
import gc

def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seeds(42)

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="",
    layout="wide"
)

class MultiBranchBrainTumorModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Branch 1: EfficientNet
        self.efficientnet = models.efficientnet_b0(pretrained=False)
        self.efficientnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.efficientnet.classifier = nn.Identity()
        
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, 512),  # This must match your saved model
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True)
        )
        
        # Branch 2: Window Attention
        self.branch2_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=4),
            nn.LayerNorm([64, 64, 64])
        )
        
        # Branch 3: Vision Transformer components
        self.patch_embed = nn.Conv2d(1, 768, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 257, 768))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512*3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Branch 1
        x1 = self.efficientnet.features(x)
        x1 = self.branch1(x1)
        
        # Branch 2
        x2 = self.branch2_conv(x)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1))
        x2 = torch.flatten(x2, 1)
        x2 = F.silu(nn.Linear(64, 512)(x2))
        
        # Branch 3
        x3 = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x3.shape[0], -1, -1)
        x3 = torch.cat((cls_token, x3), dim=1)
        x3 = x3 + self.pos_embed
        x3 = x3[:, 0]
        x3 = F.silu(nn.Linear(768, 512)(x3))
        
        # Combine branches
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.classifier(x)
        return x

class ReliableGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_handle = None
        self.backward_handle = None
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x):
        self.forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self.save_gradient)
        
        try:
            # Forward pass
            output = self.model(x)
            
            # Zero grads
            self.model.zero_grad()
            
            # Target for backprop
            pred = output.argmax(dim=1)
            one_hot = torch.zeros_like(output)
            one_hot[0][pred] = 1
            
            # Backward pass
            output.backward(gradient=one_hot)
            
            if self.gradients is None or self.activations is None:
                raise ValueError("Failed to capture gradients or activations")
            
            weights = F.adaptive_avg_pool2d(self.gradients, 1)
            
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            return cam.squeeze().cpu().numpy(), pred.item()
        
        finally:
            if self.forward_handle is not None:
                self.forward_handle.remove()
            if self.backward_handle is not None:
                self.backward_handle.remove()

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    gc.collect()
    torch.cuda.empty_cache()
    
    model = MultiBranchBrainTumorModel(num_classes=3)
    
    try:
        state_dict = torch.load('multi_branch_brain_tumor_model.pth', map_location='cpu')
        
        model_state_dict = model.state_dict()
        matched_state_dict = {}
        
        for k, v in state_dict.items():
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    matched_state_dict[k] = v
                else:
                    st.warning(f"Size mismatch for {k}: expected {model_state_dict[k].shape}, got {v.shape}")
                    if 'weight' in k:
                        nn.init.xavier_uniform_(model_state_dict[k])
                    elif 'bias' in k:
                        nn.init.zeros_(model_state_dict[k])
        
        model.load_state_dict(matched_state_dict, strict=False)
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        return model, device
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

def process_image(uploaded_file):
    try:
        if uploaded_file is None:
            return None, None
        
        image = Image.open(uploaded_file).convert('L')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0), image
    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")
        return None, None

def main():
    st.title("Brain Tumor MRI Classifier")
    
    model, device = load_model()
    if model is None:
        return
    
    uploaded_file = st.file_uploader("Upload MRI scan", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img_tensor, original_img = process_image(uploaded_file)
        if img_tensor is None or original_img is None:
            return
        
        img_tensor = img_tensor.to(device)
        
        try:
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs).item()
            
            classes = ['Glioma', 'Meningioma', 'Pituitary']
            colors = ['#FF6B6B', '#4ECDC4', '#FFD166']
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Input Image")
                st.image(original_img, caption="Original MRI", use_container_width=True)
            
            with col2:
                st.subheader("Analysis Results")
                confidence = probs[0][pred_class].item()
                st.metric("Prediction", f"{classes[pred_class]}", 
                         f"{confidence:.1%} confidence")
                
                st.write("**Class Probabilities:**")
                for i, (name, color) in enumerate(zip(classes, colors)):
                    prob = probs[0][i].item()
                    st.write(f"{name}: {prob:.4f}")
            
            # Grad-CAM Visualization
            st.subheader("Model Attention Heatmap")
            
            try:
                target_layer = model.efficientnet.features[-1]
                gradcam = ReliableGradCAM(model, target_layer)
                cam, _ = gradcam(img_tensor)
                
                if cam is not None:
                    img_np = np.array(original_img.resize((256, 256))) / 255.0
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255
                    superimposed = heatmap * 0.5 + img_np[..., np.newaxis] * 0.5
                    
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    titles = ['Original', 'Heatmap', 'Overlay']
                    images = [img_np, cam, superimposed]
                    
                    for ax, title, img in zip(axes, titles, images):
                        ax.imshow(img if title != 'Original' else img, cmap='gray' if title == 'Original' else None)
                        ax.set_title(title)
                        ax.axis('off')
                    
                    st.pyplot(fig)
            
            except Exception as e:
                st.warning(f"Grad-CAM visualization unavailable: {str(e)}")
            
            # Consistency test
            if st.checkbox("Run consistency test (predict 5 times)"):
                results = []
                for _ in range(5):
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = F.softmax(outputs, dim=1)
                        results.append(probs[0].cpu().numpy())
                
                st.write("Consistency test results (should be identical):")
                st.write(np.array(results))
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
        
        del img_tensor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()