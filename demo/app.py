import os
import sys
import yaml
import torch
import streamlit as st
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
import json
import pandas as pd
import time

# Add project root to Python path
app_dir = Path(__file__).parent
project_root = app_dir.parent
sys.path.append(str(project_root))

from src.utils import create_model
from src.data import get_transform

class ModelService:
    def __init__(self, config_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load app config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.models = {}

        # Load class mapping
        mapping_path = Path(app_dir) / 'configs' / 'class_mapping.json'
        with open(mapping_path) as f:
            mappings = json.load(f)
            self.idx_to_class = mappings['idx_to_class']
            self.num_classes = mappings['num_classes']
        
    def load_model(self, model_key):
        """Load model if not already loaded"""
        if model_key not in self.models:
            model_cfg = self.config['MODELS'][model_key]
            
            # Load model config
            config_path = app_dir / model_cfg['config_path']
            with open(config_path) as f:
                model_config = yaml.safe_load(f)
                
            # Create model
            model = create_model(model_config, self.num_classes)
            
            # Load weights
            weights_path = app_dir / model_cfg['weights_path']
            weights = torch.load(weights_path, map_location=self.device)
            if isinstance(weights, dict) and 'model' in weights:
                weights = weights['model']
            model.load_state_dict(weights)
            
            model = model.to(self.device)
            model.eval()
            self.models[model_key] = model
            
        return self.models[model_key]
    
    def predict(self, image, model_key):
        """Make prediction using selected model"""
        if image is None:
            return None
            
        model = self.load_model(model_key)
        
        try:
            # Preprocess image
            transform = get_transform((224, 224))
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                
                # Ensure output has batch dimension
                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                    
                probs = F.softmax(output, dim=1)
                pred_idx = torch.argmax(output).item()
                confidence = probs[0][pred_idx].item()
                
                # Get class name from index
                pred_class = self.idx_to_class[str(pred_idx)]
                return pred_class, confidence
                
        except Exception as e:
            raise e

@st.cache_resource
def load_service():
    """Load and cache the model service"""
    return ModelService('configs/app_config.yaml')

def main():
    # Setup streamlit page
    st.set_page_config(page_title="Multi-Model Leaf Classifier", page_icon="🍃", layout="wide")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .prediction-table {
            font-family: 'IBM Plex Sans', sans-serif;
            margin-top: 20px;
            width: 100% !important;
            table-layout: fixed;
        }
        .prediction-table th {
            background-color: #1f77b4;
            color: white;
            padding: 15px 20px;
            font-weight: 500;
            text-align: left;
            font-size: 16px;
        }
        .prediction-table td {
            padding: 12px 20px;
            border-bottom: 1px solid #eee;
            font-size: 15px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .prediction-table tr:hover {
            background-color: #f5f5f5;
        }
        /* Column widths for 4 columns */
        .prediction-table th:nth-child(1),
        .prediction-table td:nth-child(1) {
            width: 20%;
        }
        .prediction-table th:nth-child(2),
        .prediction-table td:nth-child(2) {
            width: 40%;
        }
        .prediction-table th:nth-child(3),
        .prediction-table td:nth-child(3),
        .prediction-table th:nth-child(4),
        .prediction-table td:nth-child(4) {
            width: 20%;
            text-align: right;
        }
        .model-name {
            font-weight: 500;
            color: #1f77b4;
        }
        .confidence-high {
            color: #2ecc71;
            font-weight: 500;
        }
        .confidence-medium {
            color: #f1c40f;
            font-weight: 500;
        }
        .confidence-low {
            color: #e74c3c;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🍃 Multi-Model Leaf Classifier")
    st.markdown("""
        Upload a leaf image to classify it using different models:
        - ResNet18: Classic CNN architecture
        - CvT-13: 13-layer Convolutional Vision Transformer
        - CvT-21: 21-layer Convolutional Vision Transformer 
        - ViT-B: Vision Transformer Base model
    """)

    service = load_service()
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("Model Predictions")
                results_placeholder = st.empty()
                results = []
                
                progress_containers = {
                    model_name: st.empty() 
                    for model_name in service.config['MODELS'].keys()
                }

                for model_name in service.config['MODELS'].keys():
                    try:
                        with progress_containers[model_name].container():
                            with st.spinner(f'Processing with {model_name}...'):
                                # Start timing
                                start_time = time.time()
                                
                                # Make prediction
                                label, confidence = service.predict(image, model_name)
                                
                                # Calculate processing time
                                process_time = time.time() - start_time
                                
                                # Add result
                                confidence_val = confidence * 100
                                confidence_class = (
                                    'confidence-high' if confidence_val >= 80
                                    else 'confidence-medium' if confidence_val >= 50
                                    else 'confidence-low'
                                )
                                
                                results.append({
                                    "Model": f"<span class='model-name'>{model_name}</span>",
                                    "Prediction": label.replace('_', ' ').title(),
                                    "Confidence": f"<span class='{confidence_class}'>{confidence_val:.1f}%</span>",
                                    "Time": f"{process_time:.2f}s"
                                })
                                
                                # Update results table
                                results_df = pd.DataFrame(results)
                                results_placeholder.markdown(
                                    results_df.to_html(
                                        classes='prediction-table',
                                        escape=False,
                                        index=False
                                    ),
                                    unsafe_allow_html=True
                                )
                        
                        # Clear progress container
                        progress_containers[model_name].empty()
                        
                    except Exception as e:
                        st.error(f"Error with {model_name}: {str(e)}")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
