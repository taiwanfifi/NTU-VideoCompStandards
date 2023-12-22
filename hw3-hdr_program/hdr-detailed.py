import matplotlib.pyplot as plt
import cv2
import numpy as np 

class ImageProcessor:
    def __init__(self, path):
        self.image = self.load_image(path)

    @staticmethod
    def load_image(path):
        return cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

    @staticmethod
    def tone_mapping_reinhard(hdr_image, gamma=2.2):
        tonemapReinhard = cv2.createTonemapReinhard(gamma)
        ldr_image = tonemapReinhard.process(hdr_image)
        return np.clip(ldr_image * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def adjust_local_contrast(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    @staticmethod
    def gamma_correction(image, gamma=2.2):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def process_image(self):
        if self.image is None:
            raise ValueError("Image not loaded correctly.")

        # Process HDR to SDR using Reinhard Tone Mapping
        ldr_image = self.tone_mapping_reinhard(self.image)
        contrast_image = self.adjust_local_contrast(ldr_image)
        final_image = self.gamma_correction(contrast_image)

        return final_image

    def show_image(self, image, title='Image'):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()



image_path = './hdrDebevec.hdr'  # HDR Path
processor = ImageProcessor(image_path)
processed_image = processor.process_image()

processor.show_image(processed_image, 'Processed Image')

# cv2.imshow('Processed Image', processed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

save_path = './processed_image.jpg'  # 设置保存路径
cv2.imwrite(save_path, processed_image)
print(f"Image saved at {save_path}")