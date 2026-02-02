import segmentation_models_pytorch as smp

def get_model(num_classes):
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        classes=num_classes,
        activation=None
    )
