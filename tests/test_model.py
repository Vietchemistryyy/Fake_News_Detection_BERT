#!/usr/bin/env python3
"""Quick test script to verify model loading and prediction"""

import sys
sys.path.insert(0, 'api')

from model_loader import ModelLoader
import config

def test_model():
    print("=" * 60)
    print("Testing Fake News Detection Model")
    print("=" * 60)
    
    # Test 1: Load model
    print("\n[1] Loading model...")
    ml = ModelLoader()
    success = ml.load_model()
    
    if not success:
        print(" Failed to load model")
        return False
    
    print(f" Model loaded successfully")
    print(f"  - Device: {ml.device}")
    print(f"  - Model type: {type(ml.model).__name__}")
    print(f"  - Labels: {ml.labels}")
    
    # Test 2: Simple prediction
    print("\n[2] Testing prediction...")
    test_text = "Breaking news: Scientists discover new planet in our solar system."
    
    try:
        result = ml.predict(test_text)
        print(f" Prediction successful")
        print(f"  - Text: {test_text[:50]}...")
        print(f"  - Label: {result['label']}")
        print(f"  - Confidence: {result['confidence']:.2%}")
        print(f"  - Probabilities:")
        print(f"    - Real: {result['probabilities']['real']:.2%}")
        print(f"    - Fake: {result['probabilities']['fake']:.2%}")
    except Exception as e:
        print(f" Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Another prediction
    print("\n[3] Testing another prediction...")
    test_text2 = "The government announced new economic policies to boost growth."
    
    try:
        result2 = ml.predict(test_text2)
        print(f" Prediction successful")
        print(f"  - Text: {test_text2[:50]}...")
        print(f"  - Label: {result2['label']}")
        print(f"  - Confidence: {result2['confidence']:.2%}")
    except Exception as e:
        print(f" Prediction failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print(" All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
