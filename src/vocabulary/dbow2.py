import numpy as np


class VocabularyNode:
    """
    Node in the hierarchical k-means vocabulary tree.
    """
    __slots__ = ('id', 'descriptor', 'children', 'is_leaf', 'word_id', 'weight')

    def __init__(self, node_id, descriptor):
        self.id = node_id
        self.descriptor = descriptor  # (32,) uint8 ORB descriptor
        self.children = []
        self.is_leaf = False
        self.word_id = -1
        self.weight = 0.0             # IDF weight


class VisualVocabulary:
    """
    Hierarchical Bag-of-Words visual vocabulary for binary ORB descriptors.
    Supports hierarchical tree building, descriptor-to-word quantization,
    and BowVector computation.
    """

    def __init__(self, k=10, L=3):
        self.k = int(k)               # Branching factor
        self.L = int(L)               # Depth levels
        self.root = None
        self.num_words = 0
        self.words = []               # Leaf nodes indexed by word_id

    @staticmethod
    def _hamming_distance_vectorized(des, des_array):
        """Computes Hamming distance between a single (32,) descriptor and an (N, 32) array."""
        xor_result = np.bitwise_xor(des, des_array)
        lookup = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
        return np.sum(lookup[xor_result], axis=1)

    def build_from_descriptors(self, descriptors_list):
        """
        Builds hierarchical vocabulary tree via k-means clustering on training descriptors.
        descriptors_list: list of (N, 32) uint8 arrays.
        """
        all_des = np.vstack([d for d in descriptors_list if d is not None and len(d) > 0])
        if len(all_des) < self.k:
            raise ValueError(f"Need at least {self.k} descriptors to build vocabulary.")

        node_id_counter = 0
        self.root = VocabularyNode(node_id_counter, np.zeros(32, dtype=np.uint8))
        self.words.clear()

        # Recursive hierarchical clustering
        def build_level(parent_node, des_pool, current_depth):
            nonlocal node_id_counter

            if current_depth == self.L or len(des_pool) <= self.k:
                # Create leaf visual words
                for d in des_pool:
                    leaf = VocabularyNode(node_id_counter, d)
                    node_id_counter += 1
                    leaf.is_leaf = True
                    leaf.word_id = len(self.words)
                    leaf.weight = 1.0  # Initial weight; refined by database IDF
                    self.words.append(leaf)
                    parent_node.children.append(leaf)
                return

            # K-means clustering with binary medians
            n_samples = len(des_pool)
            center_indices = np.random.choice(n_samples, min(self.k, n_samples), replace=False)
            centers = np.copy(des_pool[center_indices])

            # 3 iterations of binary k-means
            for _ in range(3):
                clusters = [[] for _ in range(len(centers))]
                for d in des_pool:
                    dists = self._hamming_distance_vectorized(d, centers)
                    best_c = int(np.argmin(dists))
                    clusters[best_c].append(d)

                for c_idx, cluster in enumerate(clusters):
                    if len(cluster) > 0:
                        bits = np.unpackbits(np.array(cluster), axis=1)
                        majority_bits = (np.mean(bits, axis=0) >= 0.5).astype(np.uint8)
                        centers[c_idx] = np.packbits(majority_bits)

            # Recurse children
            for c_idx, cluster in enumerate(clusters):
                if len(cluster) > 0:
                    child_node = VocabularyNode(node_id_counter, centers[c_idx])
                    node_id_counter += 1
                    parent_node.children.append(child_node)
                    build_level(child_node, np.array(cluster), current_depth + 1)

        build_level(self.root, all_des, 1)
        self.num_words = len(self.words)

    def transform(self, descriptors):
        """
        Quantizes (N, 32) descriptors into:
        1. BowVector: {word_id: normalized_tfidf_weight}
        2. FeatureVector: {node_id_at_level: [feature_indices]} (Direct Index)
        """
        if descriptors is None or len(descriptors) == 0 or self.root is None:
            return {}, {}

        bow_vec = {}
        feat_vec = {}

        for feat_idx, des in enumerate(descriptors):
            curr_node = self.root
            direct_node_id = curr_node.id

            # Traverse tree
            while not curr_node.is_leaf and curr_node.children:
                child_des = np.array([c.descriptor for c in curr_node.children])
                dists = self._hamming_distance_vectorized(des, child_des)
                best_idx = int(np.argmin(dists))
                curr_node = curr_node.children[best_idx]
                if not curr_node.is_leaf:
                    direct_node_id = curr_node.id

            word_id = curr_node.word_id
            weight = curr_node.weight
            bow_vec[word_id] = bow_vec.get(word_id, 0.0) + weight

            if direct_node_id not in feat_vec:
                feat_vec[direct_node_id] = []
            feat_vec[direct_node_id].append(feat_idx)

        # L1-normalization for BowVector
        norm = sum(bow_vec.values())
        if norm > 1e-7:
            for w in bow_vec:
                bow_vec[w] /= norm

        return bow_vec, feat_vec

    @staticmethod
    def score(bow1, bow2):
        """
        Computes L1 similarity score between two normalized BowVectors.
        Returns value in [0.0, 1.0] (1.0 = identical).
        """
        if not bow1 or not bow2:
            return 0.0

        common_words = set(bow1.keys()).intersection(set(bow2.keys()))
        if not common_words:
            return 0.0

        diff_sum = 0.0
        for w in common_words:
            diff_sum += abs(bow1[w] - bow2[w]) - (bow1[w] + bow2[w])

        diff_sum += 2.0  # Sum over non-common words cancels into (2.0 + diff_sum)
        score = 1.0 - 0.5 * diff_sum
        return max(0.0, min(1.0, float(score)))


class Database:
    """
    Inverted index database for fast visual place recognition candidate retrieval.
    """

    def __init__(self, vocabulary):
        self.voc = vocabulary
        # Inverted index: word_id -> dict of {keyframe_id: tfidf_weight}
        self.inverted_index = {}
        self.keyframes = {}  # kf_id -> Keyframe
        self.bow_vectors = {}  # kf_id -> BowVector

    def add(self, keyframe):
        """Adds a keyframe and updates inverted index entries."""
        if keyframe.des is None or len(keyframe.des) == 0:
            return

        bow_vec, feat_vec = self.voc.transform(keyframe.des)
        keyframe.bow_vector = bow_vec
        keyframe.feature_vector = feat_vec

        self.keyframes[keyframe.id] = keyframe
        self.bow_vectors[keyframe.id] = bow_vec

        for word_id, weight in bow_vec.items():
            if word_id not in self.inverted_index:
                self.inverted_index[word_id] = {}
            self.inverted_index[word_id][keyframe.id] = weight

    def query(self, query_bow_vector, min_common_words=5, max_results=5):
        """
        Queries the database for matching keyframes using L1-score ranking.
        Returns list of (keyframe, score) sorted by highest similarity score.
        """
        if not query_bow_vector:
            return []

        # Count shared words with historical keyframes
        candidate_counts = {}
        for word_id in query_bow_vector:
            if word_id in self.inverted_index:
                for kf_id in self.inverted_index[word_id]:
                    candidate_counts[kf_id] = candidate_counts.get(kf_id, 0) + 1

        # Filter candidates sharing enough visual words
        candidates = [kf_id for kf_id, count in candidate_counts.items() if count >= min_common_words]
        if not candidates:
            return []

        # Score candidates
        scored_candidates = []
        for kf_id in candidates:
            kf_bow = self.bow_vectors[kf_id]
            s = self.voc.score(query_bow_vector, kf_bow)
            scored_candidates.append((self.keyframes[kf_id], s))

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:max_results]