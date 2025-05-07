#!/usr/bin/env python3
"""
Test script to verify that all DeepFaceLab dependencies are working correctly.
This script imports the key libraries and performs basic operations with each.
"""

import os
import sys
import tempfile
from pathlib import Path

print("Python version:", sys.version)

# Import and test numpy
print("\n--- Testing NumPy ---")
try:
    import numpy as np
    print("NumPy version:", np.__version__)
    
    # Basic array operations
    arr = np.array([1, 2, 3, 4, 5])
    print("NumPy array creation:", arr)
    print("NumPy array mean:", np.mean(arr))
    print("NumPy array operations:", arr * 2)
    print("✓ NumPy is working correctly")
except Exception as e:
    print("✗ NumPy test failed:", str(e))

# Import and test tensorflow
print("\n--- Testing TensorFlow ---")
try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow GPU available: {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f" - {gpu}")
    else:
        print("TensorFlow GPU not available, using CPU")
    
    # Simple tensor operation
    tensor = tf.constant([[1, 2], [3, 4]])
    print("TensorFlow tensor creation:", tensor.numpy())
    print("TensorFlow tensor operation:", tf.matmul(tensor, tensor).numpy())
    print("✓ TensorFlow is working correctly")
except Exception as e:
    print("✗ TensorFlow test failed:", str(e))

# Import and test OpenCV
print("\n--- Testing OpenCV ---")
try:
    import cv2
    print("OpenCV version:", cv2.__version__)
    
    # Create a simple test image if we don't want to depend on external files
    img_size = 300
    test_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # Draw some shapes on the image
    cv2.rectangle(test_img, (50, 50), (250, 250), (0, 255, 0), 2)
    cv2.circle(test_img, (150, 150), 50, (0, 0, 255), -1)
    
    # Basic image operations
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(test_img, (15, 15), 0)
    
    print(f"Created test image shape: {test_img.shape}")
    print(f"Grayscale conversion shape: {gray_img.shape}")
    print(f"Blurred image shape: {blurred_img.shape}")
    
    # Save the test image temporarily to verify file I/O
    temp_img_path = os.path.join(tempfile.gettempdir(), "deepfacelab_test_image.jpg")
    cv2.imwrite(temp_img_path, test_img)
    print(f"Saved test image to: {temp_img_path}")
    
    # Read it back
    read_img = cv2.imread(temp_img_path)
    print(f"Read test image shape: {read_img.shape}")
    
    print("✓ OpenCV is working correctly")
except Exception as e:
    print("✗ OpenCV test failed:", str(e))

# Import and test pandas
print("\n--- Testing Pandas ---")
try:
    import pandas as pd
    print("Pandas version:", pd.__version__)
    
    # Create a simple DataFrame
    df = pd.DataFrame({
        'Name': ['John', 'Anna', 'Peter', 'Linda'],
        'Age': [28, 34, 42, 31],
        'City': ['New York', 'Paris', 'Berlin', 'London']
    })
    
    print("Pandas DataFrame creation:")
    print(df.head())
    
    # Basic operations
    print("DataFrame filtering:", df[df['Age'] > 30]['Name'].tolist())
    print("DataFrame statistics:")
    print(df.describe())
    
    print("✓ Pandas is working correctly")
except Exception as e:
    print("✗ Pandas test failed:", str(e))

# Import and test h5py
print("\n--- Testing h5py ---")
try:
    import h5py
    print("h5py version:", h5py.__version__)
    
    # Create a temporary HDF5 file
    temp_h5_path = os.path.join(tempfile.gettempdir(), "deepfacelab_test.h5")
    with h5py.File(temp_h5_path, 'w') as f:
        # Create a dataset
        dset = f.create_dataset("test_dataset", (100, 100), dtype='f')
        dset[0:10, 0:10] = np.random.random((10, 10))
        
        # Create a group
        group = f.create_group("test_group")
        group.attrs['description'] = 'This is a test group'
        
        # Dataset in the group
        group_dset = group.create_dataset("test_nested", (50, 50), dtype='i')
        group_dset[0:5, 0:5] = np.arange(25).reshape(5, 5)
        
    print(f"Created HDF5 file at: {temp_h5_path}")
    
    # Read the file back
    with h5py.File(temp_h5_path, 'r') as f:
        print("HDF5 file structure:")
        
        def print_structure(name, obj):
            print(f" - {name}: {type(obj).__name__}")
            
        f.visititems(print_structure)
        
        # Read some data
        print("Dataset sample data:")
        print(f["test_dataset"][0:3, 0:3])
        print("Group attribute:", f["test_group"].attrs['description'])
    
    print("✓ h5py is working correctly")
except Exception as e:
    print("✗ h5py test failed:", str(e))

print("\n--- Overall Test Results ---")
all_tests_passed = True

# Print a summary of all tests
for module in ['numpy', 'tensorflow', 'cv2', 'pandas', 'h5py']:
    try:
        if module in sys.modules:
            print(f"✓ {module} imported successfully")
        else:
            print(f"✗ {module} import failed")
            all_tests_passed = False
    except Exception:
        print(f"✗ Error checking {module}")
        all_tests_passed = False

if all_tests_passed:
    print("\n🎉 All tests passed! DeepFaceLab dependencies are working correctly.")
else:
    print("\n⚠️ Some tests failed. Please check the output above for details.")

