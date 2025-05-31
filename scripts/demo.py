import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import torch
if hasattr(torch, 'classes'):
    original_getattr = torch.classes.__class__.__getattr__
    def safe_getattr(self, name):
        if name == '__path__':
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return original_getattr(self, name)
    torch.classes.__class__.__getattr__ = safe_getattr
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18
import warnings

warnings.filterwarnings("ignore")

try:
    from src.models import ConvolutionalVisionTransformer, ViT
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Model specs
CVT_SPEC = {
    'cvt-13': {'NUM_STAGES': 3, 'PATCH_SIZE': [7, 3, 3], 'PATCH_STRIDE': [4, 2, 2], 'PATCH_PADDING': [2, 1, 1], 
               'DIM_EMBED': [64, 192, 384], 'DEPTH': [1, 2, 10], 'NUM_HEADS': [1, 3, 6], 'MLP_RATIO': [4.0, 4.0, 4.0], 
               'QKV_BIAS': [True, True, True], 'DROP_RATE': [0.0, 0.0, 0.0], 'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
               'DROP_PATH_RATE': [0.0, 0.0, 0.1], 'CLS_TOKEN': [False, False, True], 'POS_EMBED': [False, False, False],
               'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'], 'KERNEL_QKV': [3, 3, 3], 'PADDING_Q': [1, 1, 1], 
               'PADDING_KV': [1, 1, 1], 'STRIDE_KV': [2, 2, 2], 'STRIDE_Q': [1, 1, 1]},
    'cvt-21': {'NUM_STAGES': 3, 'PATCH_SIZE': [7, 3, 3], 'PATCH_STRIDE': [4, 2, 2], 'PATCH_PADDING': [2, 1, 1], 
               'DIM_EMBED': [64, 192, 384], 'DEPTH': [1, 4, 16], 'NUM_HEADS': [1, 3, 6], 'MLP_RATIO': [4.0, 4.0, 4.0], 
               'QKV_BIAS': [True, True, True], 'DROP_RATE': [0.0, 0.0, 0.0], 'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
               'DROP_PATH_RATE': [0.0, 0.0, 0.1], 'CLS_TOKEN': [False, False, True], 'POS_EMBED': [False, False, False],
               'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'], 'KERNEL_QKV': [3, 3, 3], 'PADDING_Q': [1, 1, 1], 
               'PADDING_KV': [1, 1, 1], 'STRIDE_KV': [2, 2, 2], 'STRIDE_Q': [1, 1, 1]}
}

@st.cache_data
def load_labels():
    try:
        with open("leafsnap_classes.txt", 'r') as f:
            return [line.strip() for line in f.readlines()]
    except:
        return []

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    
    # Load labels
    labels = load_labels()
    if not labels:
        return {}, labels, device
    
    # Model paths and loaders
    model_configs = {
        'CvT-13': ('pretrain/CvT-13-Early.pth', 'cvt'),
        'CvT-21': ('pretrain/CvT-21-Early.pth', 'cvt'),
        'ViT': ('pretrain/ViT-B-Early.pth', 'vit'),
        'ResNet-18': ('pretrain/ResNet-18.pth', 'resnet')
    }
    
    for name, (path, model_type) in model_configs.items():
        try:
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            
            # Remove module prefix
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            if model_type == 'cvt':
                head_key = next(k for k in state_dict.keys() if 'head' in k and 'weight' in k)
                num_classes = state_dict[head_key].shape[0]
                spec = 'cvt-13' if '13' in name else 'cvt-21'
                model = ConvolutionalVisionTransformer(in_chans=3, num_classes=num_classes, spec=CVT_SPEC[spec])
                
            elif model_type == 'vit':
                head_key = next(k for k in state_dict.keys() if 'head' in k and 'weight' in k)
                num_classes = state_dict[head_key].shape[0]
                model = ViT(image_size=224, patch_size=16, num_classes=num_classes, dim=768, depth=12, 
                           heads=12, dim_head=64, mlp_dim=3072, pool='cls', dropout=0.0, emb_dropout=0.1)
                           
            else:  # resnet
                model = resnet18(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
            
            model.load_state_dict(state_dict, strict=False)
            model.eval().to(device)
            models[name] = model
            
        except Exception as e:
            st.warning(f"{name}: {str(e)}")
    
    return models, labels, device

def predict(model, image, labels, device, top_k=3):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        input_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            
            probs = F.softmax(outputs, dim=-1)
            top_probs, top_indices = torch.topk(probs, min(top_k, len(labels)))
            
            return [(labels[idx], prob.item()) for idx, prob in zip(top_indices[0], top_probs[0])]
    except:
        return []

# Main app
st.set_page_config(page_title="Multi-Model Classifier", page_icon="🍃", layout="wide")
st.title("🍃 Multi-Model Image Classifier")

# Load models
models, labels, device = load_models()

if not labels:
    st.error("Could not load labels file")
    st.stop()

if not models:
    st.error("No models loaded successfully")
    st.stop()

st.success(f"Loaded {len(models)} models | {len(labels)} classes | Device: {device}")

# Upload image
uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded", use_container_width=True)
    
    with col2:
        st.subheader("Predictions")
        
        for model_name, model in models.items():
            results = predict(model, image, labels, device)
            if results:
                top_pred, confidence = results[0]
                st.write(f"**{model_name}:** {top_pred} ({confidence*100:.1f}%)")
                st.progress(confidence)
            else:
                st.error(f"{model_name}: Prediction failed")