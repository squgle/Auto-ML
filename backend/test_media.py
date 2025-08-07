#!/usr/bin/env python3
"""
Test script to check media directory permissions and model saving
"""

import os
import sys
import django
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')
django.setup()

from django.conf import settings

def test_media_directory():
    """Test if media directory can be created and written to"""
    print("Testing media directory...")
    
    # Check if MEDIA_ROOT is set
    print(f"MEDIA_ROOT: {settings.MEDIA_ROOT}")
    
    # Check if media directory exists
    if not os.path.exists(settings.MEDIA_ROOT):
        print(f"Creating media directory: {settings.MEDIA_ROOT}")
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    
    # Test models subdirectory
    models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
    print(f"Testing models directory: {models_dir}")
    
    try:
        os.makedirs(models_dir, exist_ok=True)
        print("✅ Models directory created successfully")
        
        # Test file creation
        test_file = os.path.join(models_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        print("✅ File creation test passed")
        
        # Clean up
        os.remove(test_file)
        print("✅ File cleanup test passed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_media_directory()
    if success:
        print("\n✅ Media directory test passed!")
    else:
        print("\n❌ Media directory test failed!")
        sys.exit(1) 