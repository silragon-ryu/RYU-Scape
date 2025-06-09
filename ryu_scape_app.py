import sys
import torch
import torch.nn as nn
import rasterio
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import cv2 
import torchvision
import gc
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QSlider,
    QFileDialog, QStackedLayout, QHBoxLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

# --- 1. MODEL & DATA PROCESSING LOGIC (Integrated from Notebook) ---

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- User-defined constants and functions ---
INPUT_SIZE = (256, 256)

def normalize_band(band):
    band_min, band_max = band.min(), band.max()
    if band_max > band_min:
        return (band - band_min) / (band_max - band_min)
    return band

def load_rgb(path):
    with rasterio.open(path) as src:
        b1 = normalize_band(src.read(1))
        b2 = normalize_band(src.read(2))
        b3 = normalize_band(src.read(3))
        img = np.dstack([b1, b2, b3])
    return cv2.resize(img, INPUT_SIZE)

# --- DeepLabV3 Model Setup ---
def create_deeplabv3_model(n_classes=1):
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=None)
    original_conv1 = model.backbone.conv1
    new_conv1 = nn.Conv2d(4, original_conv1.out_channels, 
                          kernel_size=original_conv1.kernel_size, 
                          stride=original_conv1.stride, 
                          padding=original_conv1.padding, 
                          bias=(original_conv1.bias is not None))
    model.backbone.conv1 = new_conv1
    model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

# --- Worker Thread for Background Processing ---
class PredictionWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, model_path, rgb_path, dsm_path):
        super().__init__()
        self.model_path = model_path
        self.rgb_path = rgb_path
        self.dsm_path = dsm_path

    def run(self):
        try:
            print("Worker Thread: Starting prediction and 3D generation.")
            model = create_deeplabv3_model(n_classes=1).to(device)
            model.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
            model.eval()

            rgb_processed = load_rgb(self.rgb_path)
            rgb_normalized_transposed = np.transpose(rgb_processed, (2, 0, 1))

            with rasterio.open(self.dsm_path) as src:
                dsm_raw = src.read(1)
            dsm_resized = cv2.resize(dsm_raw, INPUT_SIZE)
            dsm_normalized = normalize_band(dsm_resized)
            dsm_normalized = np.expand_dims(dsm_normalized, axis=0)
            
            combined_input = np.concatenate((rgb_normalized_transposed, dsm_normalized), axis=0)
            input_tensor = torch.from_numpy(combined_input).float().unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)['out']
                predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            
            del model, input_tensor, output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            confidence_threshold = 0.5 
            rgb_for_color = rgb_processed 
            z_scale = 30
            geometries = []

            print("  - Generating ground plane...")
            ground_pixels = np.where(predicted_mask <= confidence_threshold)
            ground_points = np.array([[x, -y, 0] for y, x in zip(*ground_pixels)])
            ground_colors = np.array([rgb_for_color[y, x] for y, x in zip(*ground_pixels)])
            if ground_points.size > 0:
                pcd_ground = o3d.geometry.PointCloud()
                pcd_ground.points = o3d.utility.Vector3dVector(ground_points)
                pcd_ground.colors = o3d.utility.Vector3dVector(ground_colors)
                pcd_ground = pcd_ground.voxel_down_sample(voxel_size=1.0)
                geometries.append(pcd_ground)
            
            print("  - Generating buildings with procedural details...")
            binary_building_mask = (predicted_mask > confidence_threshold).astype(np.uint8)
            contours, _ = cv2.findContours(binary_building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                mask_i = np.zeros_like(binary_building_mask)
                cv2.drawContours(mask_i, [contour], -1, 1, thickness=cv2.FILLED)
                building_pixels = np.where(mask_i == 1)
                if len(building_pixels[0]) < 10: continue

                avg_height = predicted_mask[building_pixels].mean()
                z_roof = avg_height * z_scale

                roof_points = np.array([[x, -y, z_roof] for y, x in zip(*building_pixels)])
                roof_colors = np.array([rgb_for_color[y, x] for y, x in zip(*building_pixels)])
                
                avg_wall_color = np.mean(roof_colors, axis=0) if roof_points.size > 0 else np.array([0.8, 0.8, 0.8])
                
                pcd_roof = o3d.geometry.PointCloud()
                pcd_roof.points = o3d.utility.Vector3dVector(roof_points)
                pcd_roof.colors = o3d.utility.Vector3dVector(roof_colors)
                geometries.append(pcd_roof)

                contour_points = contour.squeeze(axis=1)
                for j in range(len(contour_points)):
                    p1 = contour_points[j]; p2 = contour_points[(j + 1) % len(contour_points)]
                    
                    # Create wall with paneling texture
                    wall_vec_2d = np.array(p2) - np.array(p1)
                    wall_len = np.linalg.norm(wall_vec_2d)
                    panel_width = 8 
                    num_panels = int(wall_len / panel_width)

                    if num_panels > 0:
                        for k in range(num_panels):
                            start_p1 = p1 + (wall_vec_2d / num_panels) * k
                            end_p1 = p1 + (wall_vec_2d / num_panels) * (k + 1)
                            
                            v1 = [start_p1[0], -start_p1[1], 0]; v2 = [end_p1[0], -end_p1[1], 0]
                            v3 = [end_p1[0], -end_p1[1], z_roof]; v4 = [start_p1[0], -start_p1[1], z_roof]
                            
                            panel_vertices = [v1, v2, v3, v4]
                            panel_triangles = [[0, 1, 3], [1, 2, 3]]
                            mesh_panel = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(panel_vertices), o3d.utility.Vector3iVector(panel_triangles))
                            panel_color = avg_wall_color * (0.8 if k % 2 == 0 else 0.95)
                            mesh_panel.paint_uniform_color(panel_color)
                            geometries.append(mesh_panel)
                    else: 
                        v1 = [p1[0], -p1[1], 0]; v2 = [p2[0], -p2[1], 0]
                        v3 = [p2[0], -p2[1], z_roof]; v4 = [p1[0], -p1[1], z_roof]
                        mesh_wall_panel = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([v1,v2,v3,v4]), o3d.utility.Vector3iVector([[0,1,3],[1,2,3]]))
                        mesh_wall_panel.paint_uniform_color(avg_wall_color)
                        geometries.append(mesh_wall_panel)

                    # --- ADD WINDOWS ---
                    win_width = 8; win_height = 6; win_spacing = 12
                    num_windows = int((wall_len - win_spacing) / (win_width + win_spacing))
                    if num_windows > 0 and z_roof > (win_height * 2):
                        wall_dir_3d = np.array([wall_vec_2d[0], -wall_vec_2d[1], 0]) / wall_len if wall_len > 0 else np.array([0,0,0])
                        start_point_3d = np.array([p1[0], -p1[1], 0]) + wall_dir_3d * win_spacing

                        for k in range(num_windows):
                            win_v1 = start_point_3d + wall_dir_3d * (k * (win_width + win_spacing)) + [0,0, z_roof * 0.4]
                            win_v2 = win_v1 + wall_dir_3d * win_width
                            win_v3 = win_v2 + [0,0, win_height]
                            win_v4 = win_v1 + [0,0, win_height]
                            mesh_window = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([win_v1,win_v2,win_v3,win_v4]), o3d.utility.Vector3iVector([[0,1,3],[1,2,3]]))
                            mesh_window.paint_uniform_color([0.1, 0.1, 0.15]) # Dark blueish-gray
                            geometries.append(mesh_window)

                    # --- ADD PARAPET LIP ---
                    parapet_height = 2.0
                    para_v1 = [p1[0], -p1[1], z_roof]
                    para_v2 = [p2[0], -p2[1], z_roof]
                    para_v3 = [p2[0], -p2[1], z_roof + parapet_height]
                    para_v4 = [p1[0], -p1[1], z_roof + parapet_height]
                    mesh_parapet = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([para_v1, para_v2, para_v3, para_v4]), o3d.utility.Vector3iVector([[0, 1, 3], [1, 2, 3]]))
                    mesh_parapet.paint_uniform_color(avg_wall_color * 0.7)
                    geometries.append(mesh_parapet)

            self.finished.emit(geometries)

        except Exception as e:
            self.error.emit(f"An error occurred in the background thread: {str(e)}")


# --- 2. PyQt5 UI ---

class Ryugan3DViewer(QGLWidget):
    def __init__(self):
        super().__init__()
        self.render_data = []
        self.zoom = -150
        self.x_rot = 20
        self.y_rot = 0
        self.last_pos = None
        self.center = [128, -128, 15]

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 100, 0))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glClearColor(0.05, 0.05, 0.05, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h if h else 1, 1.0, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.x_rot, 1.0, 0.0, 0.0)
        glRotatef(self.y_rot, 0.0, 1.0, 0.0)
        glTranslatef(-self.center[0], -self.center[1], -self.center[2])

        self.draw_scene_vbo()

    def draw_scene_vbo(self):
        for item in self.render_data:
            glEnableClientState(GL_VERTEX_ARRAY)
            
            glBindBuffer(GL_ARRAY_BUFFER, item['vbo_v'])
            glVertexPointer(3, GL_FLOAT, 0, None)
            
            if item['vbo_c'] is not None:
                glEnableClientState(GL_COLOR_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, item['vbo_c'])
                glColorPointer(3, GL_FLOAT, 0, None)
            
            if item['vbo_n'] is not None:
                glEnableClientState(GL_NORMAL_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, item['vbo_n'])
                glNormalPointer(GL_FLOAT, 0, None)
            
            if item['type'] == 'points':
                glDisable(GL_LIGHTING)
                glPointSize(3.0)
                glDrawArrays(GL_POINTS, 0, item['count'])
                glEnable(GL_LIGHTING)
            elif item['type'] == 'mesh':
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, item['vbo_i'])
                glDrawElements(GL_TRIANGLES, item['count'], GL_UNSIGNED_INT, None)

            glDisableClientState(GL_VERTEX_ARRAY)
            if item['vbo_c'] is not None: glDisableClientState(GL_COLOR_ARRAY)
            if item['vbo_n'] is not None: glDisableClientState(GL_NORMAL_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            if item['type'] == 'mesh': glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def set_scene(self, geometries):
        self.cleanup_vbos()
        if not geometries: 
            self.render_data = []
            self.update()
            return

        self.center = geometries[0].get_center()

        for geom in geometries:
            if isinstance(geom, o3d.geometry.TriangleMesh):
                geom.compute_vertex_normals()

            item = {'type': None, 'vbo_v': None, 'vbo_c': None, 'vbo_n': None, 'vbo_i': None, 'count': 0}
            
            vertices = np.asarray(geom.points if isinstance(geom, o3d.geometry.PointCloud) else geom.vertices, dtype=np.float32)
            item['vbo_v'] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, item['vbo_v'])
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            
            colors = np.asarray(geom.colors if isinstance(geom, o3d.geometry.PointCloud) else geom.vertex_colors, dtype=np.float32)
            if len(colors) > 0:
                item['vbo_c'] = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, item['vbo_c'])
                glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
            
            if isinstance(geom, o3d.geometry.PointCloud):
                item['type'] = 'points'
                item['count'] = len(vertices)
            elif isinstance(geom, o3d.geometry.TriangleMesh):
                item['type'] = 'mesh'
                normals = np.asarray(geom.vertex_normals, dtype=np.float32)
                if len(normals) > 0:
                    item['vbo_n'] = glGenBuffers(1)
                    glBindBuffer(GL_ARRAY_BUFFER, item['vbo_n'])
                    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
                
                indices = np.asarray(geom.triangles, dtype=np.uint32)
                item['vbo_i'] = glGenBuffers(1)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, item['vbo_i'])
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
                item['count'] = len(indices) * 3

            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            self.render_data.append(item)

        self.update()
        
    def cleanup_vbos(self):
        for item in self.render_data:
            buffers = [b for b in [item['vbo_v'], item['vbo_c'], item['vbo_n'], item['vbo_i']] if b is not None]
            if buffers:
                glDeleteBuffers(len(buffers), buffers)
        self.render_data = []

    def wheelEvent(self, event):
        self.zoom += event.angleDelta().y() / 120.0
        self.update()

    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_pos:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            if event.buttons() & Qt.LeftButton:
                self.x_rot += dy
                self.y_rot += dx
            self.last_pos = event.pos()
            self.update()

class RyuganApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ryu-Scape (龍景)")
        self.setGeometry(100, 100, 1200, 800)
        self.rgb_path = None
        self.dsm_path = None
        
        logo_path = 'logo.png' 
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))

        self.setStyleSheet("""
            QWidget { background-color: #1a1a1a; color: #e0e0e0; font-family: 'Yu Gothic UI Light', 'Segoe UI Light', sans-serif; }
            QPushButton { background-color: #990000; color: #ffffff; font-size: 16px; padding: 12px 24px; border: 1px solid #ff3333; border-radius: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #b30000; border: 1px solid #ff5555; }
            QPushButton:pressed { background-color: #800000; }
            QPushButton:disabled { background-color: #404040; color: #888888; border: 1px solid #555555; }
            QLabel#TitleLabel { font-size: 48px; font-weight: bold; color: #ff3333; qproperty-alignment: 'AlignCenter'; }
            QLabel#SubtitleLabel { font-size: 16px; color: #a0a0a0; qproperty-alignment: 'AlignCenter'; }
            QLabel#ErrorLabel { color: #ffd700; font-weight: bold; padding: 10px; border: 2px solid #ffd700; border-radius: 8px; background-color: #332a00; }
            QFrame#ImageFrame { border: 2px dashed #ff3333; padding: 5px; border-radius: 10px; }
            QSlider::groove:horizontal { border: 1px solid #444; height: 8px; background: #333; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #ff3333; border: 1px solid #1a1a1a; width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; }
        """)

        self.stack = QStackedLayout()
        self.init_ui()
        self.setLayout(self.stack)

    def init_ui(self):
        self.init_home()
        self.init_input()
        self.init_output()

    def init_home(self):
        page = QWidget()
        layout = QVBoxLayout()
        
        logo_path = 'logo.png'
        logo_label = QLabel()
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo_label.setText("Ryu-Scape") 
            logo_label.setObjectName("TitleLabel")
        logo_label.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Cybernetic sight reveals the world's hidden dimension."); subtitle.setObjectName("SubtitleLabel")
        button = QPushButton("Initiate"); button.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        
        layout.addStretch(); 
        layout.addWidget(logo_label)
        layout.addWidget(QLabel("Ryu-Scape (龍景)"), 0, Qt.AlignCenter)
        layout.addWidget(subtitle)
        layout.addWidget(button, 0, Qt.AlignCenter)
        layout.addStretch()
        layout.setSpacing(20); page.setLayout(layout)
        self.stack.addWidget(page)

    def init_input(self):
        page = QWidget()
        layout = QVBoxLayout()
        self.rgb_label = QLabel("Upload Satellite Image"); self.rgb_label.setAlignment(Qt.AlignCenter); self.rgb_label.setFixedSize(400, 400)
        
        image_frame = QFrame(); image_frame.setObjectName("ImageFrame")
        image_layout = QHBoxLayout(); image_layout.addWidget(self.rgb_label)
        image_frame.setLayout(image_layout)
        
        upload_rgb_button = QPushButton("Load SATELLITE IMAGE"); upload_rgb_button.clicked.connect(self.load_rgb_image)
        
        self.process_button = QPushButton("GENERATE 3D MODEL"); self.process_button.clicked.connect(self.run_prediction); self.process_button.setEnabled(False)
        back_button = QPushButton("⬅ Return"); back_button.clicked.connect(lambda: self.stack.setCurrentIndex(0))

        layout.addStretch(); layout.addWidget(image_frame, 0, Qt.AlignCenter); 
        layout.addWidget(upload_rgb_button, 0, Qt.AlignCenter);
        layout.addWidget(self.process_button); layout.addWidget(back_button); layout.addStretch()
        layout.setSpacing(20); page.setLayout(layout); self.stack.addWidget(page)

    def init_output(self):
        page = QWidget()
        layout = QVBoxLayout()
        self.viewer = Ryugan3DViewer(); layout.addWidget(self.viewer, stretch=1)
        self.status_label = QLabel("3D View"); self.status_label.setAlignment(Qt.AlignCenter); layout.addWidget(self.status_label)
        back_button = QPushButton("⬅ Back"); back_button.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        layout.addWidget(back_button); layout.setSpacing(15); page.setLayout(layout); self.stack.addWidget(page)

    def load_rgb_image(self):
        start_path = "Dataset/testing/RGB"
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Satellite Image", start_path, "Images (*.png *.jpg *.tif)")
        
        if file_name:
            self.rgb_path = file_name
            self.rgb_label.setPixmap(QPixmap(file_name).scaled(400, 400, Qt.KeepAspectRatio))
            
            print(f"RGB image selected: {file_name}")
            base_filename = os.path.basename(file_name)
            
            expected_dsm_filename = f"DSM_{base_filename}"
            dsm_directory = "Dataset/testing/DSM"
            potential_dsm_path = os.path.join(dsm_directory, expected_dsm_filename)
            
            print(f"Attempting to auto-load DSM from: {potential_dsm_path}")
            if os.path.exists(potential_dsm_path):
                print("--- Corresponding DSM file found! Auto-loading. ---")
                self.dsm_path = potential_dsm_path
                self.process_button.setEnabled(True)
                self.clear_error()
            else:
                print("--- No corresponding DSM file found in testing folder. ---")
                self.show_error(f"Error: Matching DSM file not found.\nExpected: {potential_dsm_path}")
                self.process_button.setEnabled(False)
    
    def run_prediction(self):
        model_path = "Dataset/model/best_deeplabv3_model.pth"
        if not os.path.exists(model_path):
            self.show_error("Model file not found at Dataset/model/best_deeplabv3_model.pth")
            return

        self.status_label.setText("Processing... Please Wait.")
        self.stack.setCurrentIndex(2)
        QApplication.processEvents()

        self.worker = PredictionWorker(model_path, self.rgb_path, self.dsm_path)
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.error.connect(self.show_error_from_thread)
        self.worker.start()

    def on_prediction_finished(self, geometries):
        self.status_label.setText("Generated 3D Scene")
        self.viewer.set_scene(geometries)

    def show_error_from_thread(self, message):
        self.status_label.setText("Error occurred!")
        self.show_error(message)

    def show_error(self, message):
        error_label = self.findChild(QLabel, "ErrorLabel")
        if not error_label:
            error_label = QLabel(self); error_label.setObjectName("ErrorLabel")
            self.stack.widget(1).layout().insertWidget(0, error_label)
        error_label.setText(message); error_label.show()
    
    def clear_error(self):
        error_label = self.findChild(QLabel, "ErrorLabel")
        if error_label:
            error_label.hide()
            
    def closeEvent(self, event):
        # Ensure VBOs are cleaned up when the application closes
        self.viewer.cleanup_vbos()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RyuganApp()
    win.show()
    sys.exit(app.exec_())
