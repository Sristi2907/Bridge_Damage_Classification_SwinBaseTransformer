import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.model_service import model_service
from app.services.inference_service import inference_service
from app.services.xai_service import xai_service

def test_pipeline():
    print("=== Testing Backend Core and XAI Pipelines ===")
    
    # 1. Load model
    try:
        model_service.load_model()
        print("SUCCESS: Model loaded successfully.")
    except Exception as e:
        print(f"FAILED: Model loading failed: {e}")
        return False
        
    # 2. Create a dummy test image
    print("Creating dummy image...")
    dummy_img = Image.fromarray(np.uint8(np.random.randint(0, 255, (400, 400, 3))))
    
    # 3. Test prediction
    try:
        preds = inference_service.predict(dummy_img)
        print("SUCCESS: Inference completed. Results:")
        for k, v in preds.items():
            print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"FAILED: Inference failed: {e}")
        return False
        
    # 4. Test Grad-CAM overlay generation
    try:
        overlay = xai_service.generate_heatmap_overlay(dummy_img, "Crack")
        print(f"SUCCESS: Grad-CAM overlay generated. Dimensions: {overlay.size}")
        
        # Save overlay for visual verification
        os.makedirs(os.path.join("app", "core"), exist_ok=True) # just checking paths
        overlay.save("test_gradcam_overlay.jpg")
        print("SUCCESS: Saved test overlay to 'test_gradcam_overlay.jpg'")
    except Exception as e:
        print(f"FAILED: Grad-CAM overlay generation failed: {e}")
        return False
        
    print("=== Pipeline Test Completed Successfully! ===")
    return True

if __name__ == "__main__":
    test_pipeline()
