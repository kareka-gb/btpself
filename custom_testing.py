from utils import load_image
import matplotlib.pyplot as plt

def plot(img):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.title("Bicubic interpolation")
    plt.xticks([])
    plt.yticks([])

sat_img_path = f'satellite_rgb/Subset_projected_S2B_MSIL2A_20190306T052709_N0211_R105_T43RGP_20190306T101639.png'
img = load_image(sat_img_path)
img.resize((62*4, 67*4))
plot(img)
