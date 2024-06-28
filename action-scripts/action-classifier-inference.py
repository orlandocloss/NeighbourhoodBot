import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Setup argparse
parser = argparse.ArgumentParser(description='Process a text input for action classification.')
parser.add_argument('text_input', type=str, help='Text input for the model to classify.')
parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for predicting action labels (default: 0.5)')
args = parser.parse_args()

# Load the tokenizer and model
model_dir = '/home/orlando/action_saved_model'  # Update this path
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Ensure model is in evaluation mode
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to process a single text input and return the prediction
def predict(text, threshold=0.2):
    # Tokenize text input
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to probabilities
    probabilities = torch.sigmoid(outputs.logits).squeeze()
    
    # Apply threshold to get binary predictions
    predictions = (probabilities >= threshold).long()
    action_str=["Start a fundraiser","Start a petition","Start/join a group","Start a freelance/ solo venture","Start an event","Share a story"]
    
    predictions_arr= predictions.cpu().numpy()
    indices = np.where(predictions_arr == 1)[0]
    if indices.size == 0:
    	result_string = "Seek further expertise"
    else:
    	# Map these indices to the corresponding strings in the list
    	selected_strings = [action_str[index] for index in indices]

    	# Join the selected strings with commas
    	result_string = ', '.join(selected_strings)
    print(result_string)
    return(result_string)

# Use argparse to get input text and threshold
predictions = predict(args.text_input, args.threshold)
