# Introduction

This project evaluates the performance of Convolutional Vision Transformers (CvT) against traditional CNN (ResNet-18) and Vision Transformer (ViT-B) architectures. Additionally, we experiment a lightweight CvT variant for efficient leaf classification.

We provide an interactive demo application that allows you to test different models on your own leaf images:

![Demo Interface](figures\app.png)

## Dataset

The experiments are conducted on the [Leafsnap Dataset](https://www.kaggle.com/datasets/xhlulu/leafsnap-dataset), which contains:
- 185 species of trees from the Northeastern United States
- Laboratory images for training
- Field images for testing

## Model Performance

| Model    | Parameters | Accuracy |
|----------|------------|----------|
| ResNet-18| 11.7M     | 94.2%    | 
| ViT-B    | 86.1M     | 95.8%    | 
| CvT-13   | 19.3M     | 96.1%    | 
| CvT-21   | 27.6M     | 96.5%    | 
| CvT-7    | 6.8M      | 95.3%    | 

## Quick Start

### Installation
```bash
git clone https://github.com/vandat201/CvT.git
cd CvT
pip install -r requirements.txt
```

### Training
To train a model from scratch:
```bash
python tools/train.py --config configs/cvt-13-224x224.yaml \
                     --data_dir path/to/dataset \
                     --output_dir path/to/checkpoint/folder
```

### Testing
Evaluate a trained model:
```bash
python tools/test.py --config configs/cvt-13-224x224.yaml \
                    --weights path/to/weights.pth \
                    --data_dir path/to/dataset
```

### Demo Application
Run the interactive Streamlit demo:
```bash
cd demo
streamlit run app.py
```

To try the demo, download the pre-trained weights [here](https://drive.google.com/file/weights) and place them to the `demo/weights/` folder.

## Project Structure
```
├── configs/           # Model configuration files
├── demo/              # Streamlit demo 
│   ├── configs/      
│   └── weights/      
├── scripts/          # Training and testing scripts
├── src/              # Core source code
│   ├── data/         
│   ├── models/       
│   └── utils/        
├── tools/            # Helper functions
└── requirements.txt  # Project dependencies
```

## License
