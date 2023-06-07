import cv2
import numpy as np

from typing import Optional

class FourierMasker():
    def __init__(
        self,
        init_allowed_frequency: int,
        incremented_frequency: int,
        steps_per_mask: int,
        mask_threshold: int,
    ):
        self.init_allowed_frequency: int = init_allowed_frequency
        self.incremented_freqeuency: int = incremented_frequency
        self.steps_per_mask: int = steps_per_mask
        self.mask_threshold: int = mask_threshold
        self.images: Optional[np.ndarray] = None
        self.masks: Optional[np.ndarray] = None
        self.nonzero_indices: Optional[np.ndarray] = None


    def _filter(self, dft: np.ndarray, radious: int):
        rows, cols = dft.shape[:2]
        crow, ccol = rows // 2, cols // 2
        radious = max(0, radious)

        mask = np.ones((rows, cols), np.uint8)
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - crow)**2 + (y - ccol)**2 < radious**2
        mask[mask_area] = 0
        mask_dft = dft * mask
        
        inv_dft = np.fft.ifftshift(mask_dft)
        output = np.fft.ifft2(inv_dft)
        output = np.abs(output)

        return output
    
    def _fourier_transform(self, gray: np.ndarray, radious: int):
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        filtered = self._filter(dft_shift, radious)

        return filtered
    
    # def create_mask(self, images: np.ndarray, radous: int) -> np.ndarray:
    #     if len(images.shape) == 4:
    #         masks = []
    #         for image in images:
    #             gray = cv2.cvtColor(image * 255.0, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    #             filtered_image = self._fourier_transform(gray, radous)
    #             # print(filtered_image)
    #             filtered_image[filtered_image < self.mask_threshold] = 0
    #             filtered_image[filtered_image != 0] = 1
    #             masks.append(filtered_image)
    #         return np.array(masks)
    #     elif len(images.shape) == 3:
    #         gray = cv2.cvtColor(image * 255.0, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    #         filtered_image = self._fourier_transform(gray, radous)
    #         # print(filtered_image)
    #         filtered_image[filtered_image < self.mask_threshold] = 0
    #         filtered_image[filtered_image != 0] = 1
    #         return filtered_image
    #     else:
    #         return NotImplementedError(f'images should have 3 or 4 dimension, here got {len(images.shape)}')

    def create_mask(self, radous: int):
        if isinstance(self.images, np.ndarray):
            if len(self.images.shape) == 4:
                masks = []
                for image in self.images:
                    gray = cv2.cvtColor(image * 255.0, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                    filtered_image = self._fourier_transform(gray, radous)
                    # print(filtered_image)
                    filtered_image[filtered_image < self.mask_threshold] = 0
                    filtered_image[filtered_image != 0] = 1
                    masks.append(filtered_image)
                self.masks = np.array(masks)
                self.nonzero_indices = np.nonzero(self.masks) # type: ignore
            else:
                raise NotImplementedError(f'images should have 3 or 4 dimension, but got {len(self.images.shape)}')
        else:
            raise NotImplementedError(f'images should be np.ndarray type, but got {type(self.images)}')

    def mask(self, images: np.ndarray, step: int) -> np.ndarray:
        if self.images is None:
            self.images = images
        
        if step % self.steps_per_mask == 0:
            self.create_mask(self.init_allowed_frequency - step // self.steps_per_mask * self.incremented_freqeuency)

        return self.masks, self.nonzero_indices # type: ignore
