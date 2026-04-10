from PIL import Image
import numpy as np

def predict_image(image):
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0

    # Dummy logic (for demo)
    mean_val = img_array.mean()

    if mean_val > 0.5:
        return "Likely Cancerous"
    else:
        return "Likely Normal"