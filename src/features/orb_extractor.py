import cv2
import numpy as np

class ExtractorNode:
    """Represents a node in the quadtree structure for feature distribution."""

    def __init__(self, UL, UR, BL, BR):
        self.UL = UL  # Top-Left coordinate (x, y)
        self.UR = UR  # Top-Right coordinate (x, y)
        self.BL = BL  # Bottom-Left coordinate (x, y)
        self.BR = BR  # Bottom-Right coordinate (x, y)
        self.keypoints = []
        self.bNoMore = False

    def divide_node(self):
        """Splits current node into 4 equal child nodes."""
        half_x = (self.UL[0] + self.UR[0]) // 2
        half_y = (self.UL[1] + self.BL[1]) // 2

        n1 = ExtractorNode(self.UL, (half_x, self.UL[1]), (self.UL[0], half_y), (half_x, half_y))
        n2 = ExtractorNode((half_x, self.UL[1]), self.UR, (half_x, half_y), (self.UR[0], half_y))
        n3 = ExtractorNode((self.UL[0], half_y), (half_x, half_y), self.BL, (half_x, self.BL[1]))
        n4 = ExtractorNode((half_x, half_y), (self.UR[0], half_y), (half_x, self.BL[1]), self.BR)

        return n1, n2, n3, n4

class ORBExtractor:
    """
    QuadTree-based ORB Feature Extractor running strictly on CPU.
    Extracts features across scale pyramids with uniform spatial distribution.
    """

    def __init__(self, n_features=1000, scale_factor=1.2, n_levels=8, ini_th_fast=20, min_th_fast=7):
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.ini_th_fast = ini_th_fast
        self.min_th_fast = min_th_fast

        # Compute pyramid scale factors
        self.scale_factors = [1.0] * n_levels
        self.inv_scale_factors = [1.0] * n_levels
        for i in range(1, n_levels):
            self.scale_factors[i] = self.scale_factors[i - 1] * scale_factor
            self.inv_scale_factors[i] = 1.0 / self.scale_factors[i]
        
        # Allocate desired features per pyramid level
        self.features_per_level = [0] * n_levels
        factor = 1.0 / scale_factor
        desired_per_scale = n_features * (1.0 - factor) / (1.0 - np.power(factor, n_levels))
        sum_features = 0
        for i in range(n_levels - 1):
            self.features_per_level[i] = int(round(desired_per_scale))
            sum_features += self.features_per_level[i]
            desired_per_scale *= factor
        self.features_per_level[-1] = max(0, n_features - sum_features)

        # OpenCV ORB Descriptor computer instance
        self.orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=scale_factor, nlevels=n_levels)

    def extract(self, image):
        """
        Main extraction call. Input is an RGB or Grayscale CPU numpy array.
        Returns: keypoints (list of cv2.KeyPoint), descriptors (numpy array)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Compute Image Pyramid
        image_pyramid = self._compute_pyramid(gray)

        all_keypoints = []

        for level in range(self.n_levels):
            img_level = image_pyramid[level]
            desired_features = self.features_per_level[level]

            # Detect FAST corners adaptively
            fast_kps = self._detect_fast_corners(img_level)

            if len(fast_kps) == 0:
                continue

            # Distribute keypoints uniformly across image plane using QuadTree
            distributed_kps = self._distribute_quadtree(fast_kps, img_level.shape[1], img_level.shape[0], desired_features)

            # Adjust keypoint positions according to pyramid scale factor
            scale = self.scale_factors[level]
            for kp in distributed_kps:
                kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
                kp.octave = level
                kp.size = 31 * scale

            all_keypoints.extend(distributed_kps)

        # Compute ORB binary descriptors for extracted keypoints
        all_keypoints, descriptors = self.orb.compute(gray, all_keypoints)

        return all_keypoints, descriptors

    def _compute_pyramid(self, image):
        """Builds Gaussian scale pyramid."""
        pyramid = []
        current_img = image
        for level in range(self.n_levels):
            if level == 0:
                pyramid.append(current_img)
            else:
                sz = (int(round(image.shape[1] * self.inv_scale_factors[level])),
                      int(round(image.shape[0] * self.inv_scale_factors[level])))
                resized = cv2.resize(current_img, sz, interpolation=cv2.INTER_LINEAR)
                pyramid.append(resized)
        return pyramid

    def _detect_fast_corners(self, image):
        """Runs adaptive FAST detection on grid cells on CPU."""
        fast_init = cv2.FastFeatureDetector_create(threshold=self.ini_th_fast, nonmaxSuppression=True)
        kps = fast_init.detect(image, None)

        # Fallback to lower threshold if insufficient corners detected
        if len(kps) < self.features_per_level[0] // 2:
            fast_min = cv2.FastFeatureDetector_create(threshold=self.min_th_fast, nonmaxSuppression=True)
            kps = fast_min.detect(image, None)

        return kps

    def _distribute_quadtree(self, keypoints, width, height, max_features):
        """Subdivides image into quadtree nodes to ensure spatially uniform features."""
        root = ExtractorNode((0, 0), (width, 0), (0, height), (width, height))
        for kp in keypoints:
            root.keypoints.append(kp)

        nodes = [root]
        expanding_nodes = [root]

        while len(nodes) < max_features and len(expanding_nodes) > 0:
            current_nodes = list(expanding_nodes)
            expanding_nodes.clear()

            for node in current_nodes:
                if len(node.keypoints) <= 1:
                    continue

                n1, n2, n3, n4 = node.divide_node()

                for kp in node.keypoints:
                    x, y = kp.pt
                    if x < n1.UR[0]:
                        if y < n1.BL[1]:
                            n1.keypoints.append(kp)
                        else:
                            n3.keypoints.append(kp)
                    else:
                        if y < n2.BL[1]:
                            n2.keypoints.append(kp)
                        else:
                            n4.keypoints.append(kp)

                nodes.remove(node)
                for child in [n1, n2, n3, n4]:
                    if len(child.keypoints) > 0:
                        nodes.append(child)
                        if len(child.keypoints) > 1:
                            expanding_nodes.append(child)

        result_kps = []
        for node in nodes:
            if len(node.keypoints) > 0:
                # Retain keypoint with highest response in each quadrant
                best_kp = max(node.keypoints, key=lambda k: k.response)
                result_kps.append(best_kp)

        return result_kps