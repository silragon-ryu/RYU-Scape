import sys
from PyQt5.QtWidgets import QApplication

# We need to import the application class from your main script.
# This assumes your main script is named 'ryu_scape_app.py'.
from ryu_scape_app import RyuganApp

def test_app_initialization():
    """
    This is a simple "smoke test" to ensure the main application
    window can be created without raising an immediate error.
    
    It checks that all the UI components are set up correctly.
    """
    
    # Pytest needs a QApplication instance to be able to create Qt widgets.
    # We create one here for the scope of this test.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    print("Creating instance of RyuganApp for testing...")
    try:
        # Attempt to create the main window.
        window = RyuganApp()
        # If the window is created successfully, the test passes.
        assert window is not None
        print("RyuganApp instance created successfully.")
    except Exception as e:
        # If any exception occurs during initialization, fail the test.
        assert False, f"Failed to initialize RyuganApp: {e}"

# You can add more tests below. For example, testing specific functions.
# def test_normalize_band_function():
#     from ryu_scape_app import normalize_band
#     import numpy as np
#     # Create a sample numpy array
#     test_band = np.array([0, 50, 100], dtype=np.float32)
#     normalized = normalize_band(test_band)
#     # Check if the output is what we expect
#     assert np.allclose(normalized, [0.0, 0.5, 1.0])
#     print("normalize_band() test passed.")

