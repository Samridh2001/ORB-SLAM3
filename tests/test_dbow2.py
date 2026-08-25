import os
import sys
import numpy as np
import cv2

# Dynamic path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datastructures.frame import Frame
from src.datastructures.keyframe import Keyframe
from src.vocabulary.dbow2 import VisualVocabulary, Database


def test_dbow2_system():
    np.random.seed(42)

    # 1. Generate synthetic training descriptors for Vocabulary (branching=4, depth=2 -> ~16 words)
    voc = VisualVocabulary(k=4, L=2)
    training_data = [np.random.randint(0, 256, (50, 32), dtype=np.uint8) for _ in range(5)]
    voc.build_from_descriptors(training_data)

    print(f"Built Vocabulary with {voc.num_words} visual words.")
    assert voc.num_words > 0, "Failed to build visual vocabulary!"

    # 2. Test BowVector Transformation & Normalization
    query_des = np.random.randint(0, 256, (40, 32), dtype=np.uint8)
    bow_vec, feat_vec = voc.transform(query_des)

    assert len(bow_vec) > 0, "BowVector transformation returned empty dictionary!"
    assert np.isclose(sum(bow_vec.values()), 1.0, atol=1e-5), "BowVector was not L1-normalized!"

    # Test Self-Similarity Score
    self_score = voc.score(bow_vec, bow_vec)
    print(f"Self-Similarity L1 Score: {self_score:.4f} (Expected: 1.0000)")
    assert np.isclose(self_score, 1.0, atol=1e-3), "Self-score must be 1.0!"

    # 3. Test Inverted Index Database Querying
    db = Database(voc)

    # Create 3 distinct keyframes
    des_kf0 = np.random.randint(0, 256, (60, 32), dtype=np.uint8)
    des_kf1 = np.random.randint(0, 256, (60, 32), dtype=np.uint8)
    # KF2 is near identical to KF0 (loop scenario)
    des_kf2 = np.copy(des_kf0)
    des_kf2[0:5] = np.random.randint(0, 256, (5, 32), dtype=np.uint8)  # slight noise

    kf0 = Keyframe(Frame([cv2.KeyPoint(10, 10, 31)], des_kf0, timestamp=0.0))
    kf1 = Keyframe(Frame([cv2.KeyPoint(20, 20, 31)], des_kf1, timestamp=1.0))
    kf2 = Keyframe(Frame([cv2.KeyPoint(30, 30, 31)], des_kf2, timestamp=2.0))

    db.add(kf0)
    db.add(kf1)

    # Query with KF2 (should match KF0 as top place recognition candidate)
    query_bow, _ = voc.transform(kf2.des)
    results = db.query(query_bow, min_common_words=2, max_results=2)

    assert len(results) > 0, "Database query returned no candidate matches!"
    best_match_kf, best_score = results[0]
    print(f"Top Place Recognition Candidate: KF #{best_match_kf.id} with Score: {best_score:.4f}")

    assert best_match_kf.id == kf0.id, "Place recognition matched wrong keyframe!"
    assert best_score > 0.6, "Loop match score should be high for similar descriptor sets!"

    print("[TEST DBOW2 VISUAL VOCABULARY SUCCESS]")


if __name__ == "__main__":
    test_dbow2_system()