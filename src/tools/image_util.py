import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import torch

def denormalize(image, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return image * std + mean

def view_img(tensor_image, title = "title"):
    img_data = tensor_image.numpy()

    # 이미지를 시각화 해보자
    plt.imshow(np.transpose(img_data, (1, 2, 0)))
    plt.title(title)
    plt.show()