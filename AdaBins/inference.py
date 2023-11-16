from infer import InferenceHelper
from PIL import Image
import matplotlib.pyplot as plt

"""
predictions using nyu dataset
"""

infer_helper = InferenceHelper(dataset='kitti')

# predict depth of a single pillow image
img = Image.open("test_imgs/casd1.png").resize((640,480)).convert("RGB")  # any rgb pillow image

bin_centers, predicted_depth = infer_helper.predict_pil(img)

plt.imshow(predicted_depth[0][0], cmap='plasma')
plt.colorbar()
plt.show()