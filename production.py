import matplotlib.pyplot as plt
from functions import *
import pickle

def main():
    fig, axs = plt.subplots(2, 5, figsize=(24,10))
    axs = axs.flatten()

    for i in range(10):
        img_gray, img_rgb = prepare_img(f'images/img{i}.png')
        circles = find_circles(img_gray)
        circles = treat_circles(circles)

        if circles is not None:
            filtered_circles = filter_inner_circles(circles, 10)
            img_rgb = draw_circles(img_rgb, filtered_circles)

        axs[i].imshow(img_rgb, cmap='gray')
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
