import torch
from datasets import load_dataset
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification


def EfficentNetB4(**kwargs):
        # Unpack kwargs
    X_train = kwargs['X_train']
    y_train = kwargs['y_train']
    X_test = kwargs['X_test']
    y_test = kwargs['y_test']
    
    image = X_train["X_train"]["image"][0]

    preprocessor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b4")
    model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b4")

    inputs = preprocessor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])
