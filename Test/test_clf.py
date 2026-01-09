import sys
import traceback
import os

# Add project root to path
sys.path.insert(0, ".")

try:
    print("Starting test wrapper...")
    # Import main module
    import model.train_classifier as tc
    
    # Run main function
    tc.train_dual_classifiers()
    
except Exception as e:
    print("\nCRASH DETECTED!")
    print(f"Error: {e}")
    traceback.print_exc()
