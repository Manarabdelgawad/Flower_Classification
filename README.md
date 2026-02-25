# Flower Classification Web App

A complete Deep Learning project for multi-class flower classification using Transfer Learning (ResNet50) 


## Project Overview

This project classifies flower images into 5 categories:

Daisy

Dandelion

Rose

Sunflower

Tulip

The model is trained using transfer learning with a pretrained CNN model and deployed using a production-ready API.

## Model Architecture

Pretrained ResNet50 (ImageNet weights)

GlobalAveragePooling layer

Dropout (Regularization)

Dense Softmax classifier

Loss: Categorical Crossentropy

Optimizer: Adam

##  Installation

1. clone project
```bash
git clone <your-repo-url>
cd flower_classification
```
2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
3. install requirements
```bash
pip install -r requirements.txt
```
## Run the Project

1. Train the Model
```bash
python train.py
```
2. Evaluate Model
```bash
python evaluate.py
```
3. Run FastAPI Backend
```bash
uvicorn api:app --reload
```
4. test API directly from Swagger UI
```bash
http://127.0.0.1:8000/docs
```
5. Run Streamlit App
```bash
streamlit run app.py
```


