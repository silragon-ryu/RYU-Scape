# 🐉 Ryu-Scape (龍景)

**Cybernetic sight reveals the world's hidden dimension.**

Ryu-Scape is a desktop application that leverages deep learning to transform 2D satellite imagery into richly detailed 3D scenes. By combining RGB satellite images with their corresponding Digital Surface Models (DSM), Ryu-Scape reconstructs interactive, textured, and procedurally detailed environments.

---

## 📸 Screenshots

_Add screenshots or a GIF showing your application in action here!_

---

## ✨ Features

- **🧠 AI-Powered Segmentation**  
  Uses a DeepLabV3 model with a ResNet-50 backbone to detect building footprints from satellite images.

- **📂 Automated DSM Handling**  
  Automatically locates and loads the corresponding DSM file for the selected RGB image.

- **🧱 Advanced 3D Generation**  
  - Textured ground plane using original satellite image  
  - Extruded 3D buildings with real-world height  
  - Procedural details like windows, walls, and rooftop elements

- **🌀 Interactive 3D Viewer**  
  Built with OpenGL for smooth real-time scene manipulation (rotate, zoom, pan)

- **🎨 Custom UI Theme**  
  Inspired by Japanese cyberpunk, built with PyQt5.

- **📦 Standalone Executable**  
  Easily packageable for Windows with PyInstaller — no Python installation needed.

---

## 🛠️ Technologies Used

| Category            | Tools / Libraries                             |
|---------------------|-----------------------------------------------|
| Language            | Python 3                                      |
| Deep Learning       | PyTorch                                       |
| Computer Vision     | OpenCV                                        |
| Geospatial Handling | Rasterio                                      |
| 3D Rendering        | Open3D, PyOpenGL                              |
| GUI Framework       | PyQt5                                         |
| Packaging           | PyInstaller                                   |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/silragon-ryu/RYU-Scape.git
cd RYU-Scape
