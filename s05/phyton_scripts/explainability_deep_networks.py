import grelu
from torchvision.models import resnet50, ResNet50_Weights
import torch
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path: str, weights: ResNet50_Weights) -> torch.Tensor:
    preprocess = weights.transforms()
    # preprocess:
    # ImageClassification(
    #     crop_size=[224]
    #     resize_size=[256]
    #     mean=[0.485, 0.456, 0.406]
    #     std=[0.229, 0.224, 0.225]
    #     interpolation=InterpolationMode.BILINEAR
    # )

    img = Image.open(image_path).convert("RGB")

    #.unsqueeze(0) -> dim is (1, 3, 224, 224)
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor

def compute_guided_grads(model, X, target_class=243) -> torch.Tensor:
    X.requires_grad_()
    output = model(X)  # dim [1, 1000] 1000 ImageNet classes one logic for each
    model.zero_grad() # reset grads
    score = output[0, target_class] # 1 image, select "bull mastiff" with index 243
    score.backward() # backpropagation, i.e. grad calculation

    gradients = X.grad.data.squeeze(0) # remove batch dimension -> dim is [3, 224, 224]
    return gradients

def plot_grads(grads: torch.Tensor) -> None:
    # dim (224, 224), saliency [de: Ausprägung]
    saliency = grads.abs().max(dim=0)[0]

    # Normalize to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    saliency = saliency.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(saliency, cmap='gray')
    plt.title("Guided Backpropagation Saliency Map")
    plt.axis('off')
    plt.show()

def run_all(image_path: str) -> None:
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    grelu.replace_relu_with_guided(model)
    model.eval()
    img_tensor = load_and_preprocess_image(image_path, weights)
    grads = compute_guided_grads(model, img_tensor, target_class=35)
    plot_grads(grads)



if __name__ == "__main__":
    IMAGE_PATH = "test_imag.png"
    run_all(IMAGE_PATH)