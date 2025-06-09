🐉 Ryu-Scape (龍景)
Cybernetic sight reveals the world's hidden dimension.

Ryu-Scape is a desktop application that leverages a deep learning model to convert 2D satellite imagery into detailed 3D scenes. By providing an RGB satellite image and its corresponding Digital Surface Model (DSM), the application generates an interactive 3D model, complete with extruded buildings, procedural details, and a textured ground plane.

📸 Screenshots
(Here you can add a screenshot of your application in action. A GIF showing the 3D interaction would be even better!)

✨ Features
AI-Powered Segmentation: Utilizes a trained DeepLabV3 model with a ResNet-50 backbone to accurately identify building footprints from satellite images.

Automated Workflow: Automatically finds and loads the required Digital Surface Model (DSM) file based on the selected RGB image, streamlining the user experience.

Advanced 3D Generation:

Creates a textured ground plane using the original satellite image.

Extrudes building footprints into solid 3D models with realistic height.

Procedurally generates architectural details like textured walls, rooftop parapets, and windows to enhance realism.

Interactive 3D Viewer: A custom high-performance OpenGL viewer that allows for smooth rotation, panning, and zooming of the generated scene.

Custom Theming: Features a unique "Cyberpunk/Japanese" themed user interface built with PyQt5.

Standalone Executable: Can be packaged into a single folder for easy distribution on Windows, with no need for a local Python installation.

🛠️ Technologies Used
Programming Language: Python 3

Deep Learning: PyTorch

Computer Vision: OpenCV

Geospatial Data: Rasterio

3D Graphics & Processing: Open3D, PyOpenGL

GUI Framework: PyQt5

Packaging: PyInstaller

⚙️ Setup and Installation
To run this project from the source code, please follow these steps:

1. Clone the Repository:

git clone https://github.com/your-username/ryu-scape.git
cd ryu-scape

1a. Set Up Git LFS for Large File Storage

This project uses Git LFS to manage the large model file (.pth). You must have Git LFS installed to clone the model file correctly.

Install Git LFS:

Windows: Download and run the installer from git-lfs.github.com.

macOS: brew install git-lfs

Linux (Debian/Ubuntu): sudo apt-get install git-lfs

Set up Git LFS in the repository:
After installing, run this command once for your user account:

git lfs install

When you clone the repository, Git LFS will automatically download the large model file.

(For contributors) If you were adding a new large file, you would track it using:

git lfs track "*.pth"

2. Create a Python Virtual Environment:

It is highly recommended to use a virtual environment to manage dependencies.

# For Windows
python -m venv .venv
.\.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies:

The required libraries are listed in requirements.txt. Install them using pip.

pip install -r requirements.txt

(Note: You will need to create a requirements.txt file. You can generate one from your current environment by running pip freeze > requirements.txt)

🚀 Usage
There are two ways to run the Ryu-Scape application:

1. From Source Code:

Once you have completed the setup and installation, simply run the main application script:

python ryu_scape_app.py

2. From the Executable (for Windows):

An executable version can be created using PyInstaller.

Follow the instructions in the Packaging Guide.

Navigate to the dist/Ryu-Scape folder.

Double-click on Ryu-Scape.exe to launch the application.

📁 Project Structure
ryu-scape/
│
├── ryu_scape_app.py      # Main application script
├── logo.png              # Application logo
├── requirements.txt      # List of Python dependencies
├── .gitattributes        # Git LFS tracking file
│
└── Dataset/
    ├── model/
    │   └── best_deeplabv3_model.pth    # Trained AI model (handled by Git LFS)
    │
    └── testing/
        ├── RGB/
        │   └── (RGB satellite images...)
        │
        └── DSM/
            └── (Corresponding DSM files...)

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
