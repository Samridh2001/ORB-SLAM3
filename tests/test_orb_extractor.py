import numpy as np
import cv2
from src.features.orb_extractor import ORBExtractor

def test_extractor():
    # Generate synthetic CPU noise image
    dummy_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    # Add artificial high-contrast features
    cv2.rectangle(dummy_img, (100, 100), (200, 200), 255, -1)
    cv2.circle(dummy_img, (400, 300), 50, 0, -1)

    extractor = ORBExtractor(n_features=500)
    kps, des = extractor.extract(dummy_img)

    print(f"[TEST SUCCESS] Extracted {len(kps)} keypoints on CPU.")
    if des is not None:
        print(f"Descriptors Matrix Shape: {des.shape}")

if __name__ == "__main__":
    test_extractor()