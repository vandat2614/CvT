from torchvision import transforms
from PIL import Image

def resize_image(image, output_size):
    """Resize image preserving aspect ratio with black padding
    
    Args:
        image (PIL.Image): Input image
        output_size (tuple): Target size (width, height)
        
    Returns:
        PIL.Image: Resized and padded image
    """
    # Calculate ratio to preserve aspect ratio
    width, height = image.size
    ratio = min(output_size[0] / width, output_size[1] / height)
    new_size = (int(width * ratio), int(height * ratio))

    # Resize the image
    resized_img = image.resize(new_size, Image.LANCZOS)

    # Create new image with black background
    new_img = Image.new("RGB", output_size, (0, 0, 0))

    # Calculate position to center the resized image
    left = (output_size[0] - new_size[0]) // 2
    top = (output_size[1] - new_size[1]) // 2

    # Paste resized image onto background
    new_img.paste(resized_img, (left, top))
    return new_img

class Resize:
    """Transform to resize image while preserving aspect ratio"""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if img.size == self.size:
            return img
        return resize_image(img, self.size)

def get_transform(img_size=(224, 224)):
    """Get image transforms for training/testing
    
    Args:
        img_size (tuple): Target image size (height, width)
        is_training (bool): Whether to use training transforms
        
    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    return transforms.Compose([
        Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])