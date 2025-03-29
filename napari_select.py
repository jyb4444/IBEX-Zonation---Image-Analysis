import napari
import numpy as np
import json
try:
    import nrrd
except ImportError:
    try:
        import pynrrd as nrrd
        print("Using pynrrd instead of nrrd")
    except ImportError:
        print("Error: Cannot import nrrd or pynrrd library")
        exit(1)
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from skimage import io as skio  
import os
from napari.utils.colormaps import DirectLabelColormap

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  

# Load the NRRD file
filename = "./PORTAL_CENTRAL_FULL_SECTION_20250131.nrrd"
image, header = nrrd.read(filename)

# Convert image to native byte order
image = image.astype(np.float32, copy=True)  # This ensures native byte order
# print(image.shape)
# print(image[10000, 10000, :])

# 读取tissue boundary mask
try:
    # 解除skimage的图像大小限制
    os.environ['SKIMAGE_ALLOW_HUGE_IMAGES'] = '1'
    
    # 检查文件是否存在
    if os.path.exists("./image/largest_contour_mask.png"):
        tissue_mask = skio.imread("./image/largest_contour_mask.png")
        # 确保mask是二值的
        if len(tissue_mask.shape) > 2:  # 如果是RGB图像
            tissue_mask = tissue_mask[:, :, 0] > 0  # 只取一个通道并转为二值
        else:
            tissue_mask = tissue_mask > 0
        
        # 确保mask与图像尺寸匹配
        if tissue_mask.shape[:2] != image.shape[:2]:
            print(f"警告: Mask尺寸 {tissue_mask.shape[:2]} 与图像尺寸 {image.shape[:2]} 不匹配")
            # 如果尺寸不匹配，可以考虑调整mask的大小
            from skimage.transform import resize
            tissue_mask = resize(tissue_mask, image.shape[:2], order=0, preserve_range=True).astype(bool)
            print(f"已调整mask尺寸为 {tissue_mask.shape}")
        
        # 应用mask到图像上
        # 对于3通道图像
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                # 只保留mask内的像素，其余设为背景值
                background_value = np.min(image[:, :, i])
                temp = image[:, :, i].copy()
                temp[~tissue_mask] = background_value
                image[:, :, i] = temp
        # 对于单通道图像        
        else:
            background_value = np.min(image)
            image[~tissue_mask] = background_value
        
        print("成功应用tissue boundary mask")
    else:
        print("警告: largest_contour_mask.png 文件不存在")
except Exception as e:
    print(f"应用tissue boundary mask时发生错误: {e}")
    print("继续执行程序，但不应用mask")

# Define tile size
TILE_SIZE = 6000

# Function to get a random tile
def get_random_tile(image, tile_size):
    max_x, max_y, _ = image.shape
    x = random.randint(0, max_x - tile_size)
    y = random.randint(0, max_y - tile_size)
    return image[x:x + tile_size, y:y + tile_size], (x, y)

# Initialize Napari viewer
viewer = napari.Viewer()

# Get initial tile
tile, (x_offset, y_offset) = get_random_tile(image, TILE_SIZE)
tile_layer = viewer.add_image(tile, name="Current Tile")

# Add a Labels layer for contour annotation
labels_layer = viewer.add_labels(
    np.zeros(tile.shape[:2], dtype=np.uint32),
    name='Contour Annotation'
)

# 设置Contour Annotation图层的优先级更高
# 确保标注层在图像层之上，便于观察
labels_layer.blending = 'additive'  # 使用加性混合让标注更明显
labels_layer.opacity = 0.8          # 设置不透明度为80%

# 设置标签层的颜色 - 修复颜色映射问题
labels_layer.brush_size = 10  # 调整默认画笔大小

# 创建一个从1开始的颜色映射，其中1对应亮黄色
colors = {1: [255, 0, 255, 255]}  # 亮黄色，带有完全不透明的alpha通道
labels_layer.color = colors

# For storing all annotations
annotations = []
current_annotation = None

# Flag to track if labels have changed
label_changed = False

# 调试功能：展示保存的mask内容
def debug_show_mask(mask_file):
    try:
        mask_data = np.load(mask_file)
        viewer.add_labels(mask_data, name=f"Debug: {mask_file}")
        print(f"Loaded mask for debugging: {mask_file}")
        print(f"Mask shape: {mask_data.shape}, Unique values: {np.unique(mask_data)}")
    except Exception as e:
        print(f"Error loading mask for debug: {e}")

# Monitor changes to the labels layer
def on_labels_change(event):
    global label_changed
    label_changed = True
    print("Annotation updated!")
    print(f"Current annotation has {np.sum(labels_layer.data > 0)} labeled pixels")

# Register callback for label changes
labels_layer.events.data.connect(on_labels_change)

# Function to load the next image
def load_next_image():
    global tile, x_offset, y_offset, tile_layer, labels_layer, label_changed, current_annotation
    
    # Save current tile annotations
    # save_current_annotations()
    
    # 清理临时可视化层
    for layer in list(viewer.layers):
        if layer.name.startswith("Current Annotations Debug") or layer.name.startswith("Debug: mask_"):
            viewer.layers.remove(layer)
    
    # Get a new random tile
    tile, (x_offset, y_offset) = get_random_tile(image, TILE_SIZE)
    
    # Update the image in the viewer
    tile_layer.data = tile
    
    # 尝试选择图像层作为活动层 - 使用正确的API
    try:
        # 在较新版本的napari中使用
        viewer.layers.selection.active = tile_layer
    except Exception as e:
        try:
            # 在某些版本的napari中使用
            viewer.layers.selection.select_only(tile_layer)
        except Exception as e2:
            print(f"注意: 无法将图像层设置为活动层: {e2}")
    
    print(f"Loaded new tile, position: ({x_offset}, {y_offset})")
    
    # Clear annotations - 创建新的空白数组
    new_labels = np.zeros(tile.shape[:2], dtype=np.uint32)
    labels_layer.data = new_labels
    
    # 刷新标注层 - 尝试强制更新显示
    temp = labels_layer.selected_label
    labels_layer.selected_label = 1
    labels_layer.selected_label = temp
    
    # 确保labels_layer被正确更新
    print(f"重置标注层，形状: {labels_layer.data.shape}")
    if np.any(labels_layer.data > 0):
        print("警告: 标注层重置失败，仍有标注数据")
    
    # Reset flags
    label_changed = False
    current_annotation = {
        "tile_position": (int(x_offset), int(y_offset)),
        "tile_size": TILE_SIZE,
        "has_label_mask": False,
        "timestamp": str(np.datetime64('now'))
    }

def save_current_annotations():
    global labels_layer, x_offset, y_offset, annotations, label_changed, current_annotation
    
    # If no current annotation info, create one
    if current_annotation is None:
        current_annotation = {
            "tile_position": (int(x_offset), int(y_offset)),
            "tile_size": TILE_SIZE,
            "has_label_mask": False,
            "timestamp": str(np.datetime64('now'))
        }
    
    # Check if labels layer has annotations
    labels_data = labels_layer.data
    has_labels = np.any(labels_data > 0)
    
    # 输出当前标注信息用于调试
    print(f"Labels data type: {labels_data.dtype}, shape: {labels_data.shape}")
    print(f"Contains annotations: {has_labels}, unique values: {np.unique(labels_data)}")
    
    # If there are contour annotations
    if has_labels:
        # Save label data to a file
        mask_filename = f"./image/mask_{len(annotations)}.npy"
        np.save(mask_filename, labels_data)
        
        # 检查保存是否成功
        if os.path.exists(mask_filename):
            print(f"成功保存mask到文件: {mask_filename}")
            
            # 检查保存的文件内容
            try:
                saved_mask = np.load(mask_filename)
                print(f"验证保存的mask: 形状={saved_mask.shape}, 最大值={np.max(saved_mask)}, 标记像素数={np.sum(saved_mask > 0)}")
            except Exception as e:
                print(f"验证保存的mask时出错: {e}")
        else:
            print(f"警告: 文件{mask_filename}未能成功创建!")
            
        current_annotation["has_label_mask"] = True
        current_annotation["mask_file"] = mask_filename
        print(f"Contour annotation saved to {mask_filename}")
        
        # 调试：显示保存的mask
        debug_show_mask(mask_filename)
        
        # Add to annotations list
        annotations.append(current_annotation)
        print(f"Saved current tile annotation with contours")
        
        # Update the annotations.json file
        with open("annotations.json", "w") as f:
            json.dump(annotations, f)
        print(f"All annotations saved to annotations.json, total: {len(annotations)} images")
    else:
        print("No annotations found to save")

def train_model():
    global model, optimizer, annotations, image
    
    print("Starting model training...")
    
    if not annotations:
        print("警告：没有找到任何标注数据。请先进行一些标注。")
        return
    
    try:
        # Prepare training data
        # Create mask image from annotations
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        
        # 统计共有多少个标注
        print(f"正在处理 {len(annotations)} 个标注...")
        
        loaded_masks = 0
        
        # Convert annotations to mask
        for i, anno in enumerate(annotations):
            # Process contour annotations
            if anno.get("has_label_mask", False) and "mask_file" in anno:
                mask_file = anno["mask_file"]
                
                # 检查文件是否存在
                if not os.path.exists(mask_file):
                    print(f"警告: 找不到mask文件 {mask_file}")
                    continue
                    
                try:
                    print(f"正在加载mask文件 {mask_file}...")
                    label_mask = np.load(mask_file)
                    print(f"成功加载mask，形状: {label_mask.shape}, 唯一值: {np.unique(label_mask)}")
                    
                    x_pos, y_pos = anno["tile_position"]
                    tile_size = anno["tile_size"]
                    
                    print(f"标注位置: ({x_pos}, {y_pos}), 大小: {tile_size}")
                    
                    # Ensure we don't exceed boundaries
                    x_end = min(image.shape[0], x_pos + tile_size)
                    y_end = min(image.shape[1], y_pos + tile_size)
                    
                    # Calculate actual mask size
                    mask_height = x_end - x_pos
                    mask_width = y_end - y_pos
                    
                    print(f"应用范围: ({x_pos}:{x_end}, {y_pos}:{y_end})")
                    print(f"实际大小: 高度={mask_height}, 宽度={mask_width}")
                    
                    # 确保标签大小与应用区域匹配
                    actual_mask = label_mask
                    if label_mask.shape[0] > mask_height or label_mask.shape[1] > mask_width:
                        actual_mask = label_mask[:mask_height, :mask_width]
                        print(f"调整mask大小以适应目标区域")
                    
                    # Copy mask to global mask
                    mask[x_pos:x_end, y_pos:y_end] = np.logical_or(
                        mask[x_pos:x_end, y_pos:y_end],
                        actual_mask > 0
                    ).astype(np.float32)
                    
                    loaded_masks += 1
                    print(f"已应用mask {i+1}/{len(annotations)}")
                    
                except Exception as e:
                    print(f"Error loading mask file {mask_file}: {e}")
                    import traceback
                    traceback.print_exc()
        
        if loaded_masks == 0:
            print("错误: 没有成功加载任何mask文件。无法继续训练。")
            return
            
        print(f"成功加载了 {loaded_masks}/{len(annotations)} 个mask文件")
        
        # 检查全局mask
        mask_pixels = np.sum(mask > 0)
        print(f"全局mask统计: 形状={mask.shape}, 标记像素数={mask_pixels}")
        
        if mask_pixels == 0:
            print("错误: 合并后的全局mask为空。没有任何标注数据可以用于训练。")
            return
        
        # 显示全局mask供参考
        viewer.add_image(mask, name="Global Mask")
        
        # Convert to PyTorch tensors - with explicit conversion to float32
        print("Converting image to tensor...")
        # Create a single-channel version of the image for training
        single_channel_image = np.mean(image, axis=2, dtype=np.float32)
        x_train = torch.tensor(single_channel_image[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
        y_train = torch.tensor(mask[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
        
        print(f"Tensor shapes - Input: {x_train.shape}, Target: {y_train.shape}")
        
        # Create dataset and dataloader
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Train the model
        model.train()
        for epoch in range(5):
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = F.mse_loss(output, y_batch)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
            print(f"Completed training epoch {epoch+1}/5")
        
        print("Model training completed")
        
        # Apply model to the entire image
        apply_model_to_image()
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

def apply_model_to_image():
    global model, image, viewer
    
    print("Applying model to the entire image...")
    
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Convert image to single channel for prediction
        single_channel_image = np.mean(image, axis=2, dtype=np.float32)
        
        # Generate predictions
        with torch.no_grad():
            # Process in chunks if the image is too large
            chunk_size = 1024
            height, width = single_channel_image.shape
            prediction = np.zeros((height, width), dtype=np.float32)
            
            for y in range(0, height, chunk_size):
                for x in range(0, width, chunk_size):
                    # Define chunk boundaries
                    y_end = min(y + chunk_size, height)
                    x_end = min(x + chunk_size, width)
                    
                    # Extract chunk
                    chunk = single_channel_image[y:y_end, x:x_end]
                    
                    # Convert to tensor
                    chunk_tensor = torch.tensor(chunk[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
                    
                    # Predict
                    chunk_pred = model(chunk_tensor).squeeze().numpy()
                    
                    # Insert prediction back
                    prediction[y:y_end, x:x_end] = chunk_pred
                    
                    print(f"Processed chunk at ({y}, {x})")
            
            # Add prediction results to viewer
            viewer.add_image((prediction > 0.5).astype(float), name="Model Prediction")
            
            print("Prediction completed and displayed")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

# 添加一个查看当前标注的按钮
def view_current_annotations():
    global labels_layer
    
    # 显示当前标注的信息
    labels_data = labels_layer.data
    unique_values = np.unique(labels_data)
    labeled_pixels = np.sum(labels_data > 0)
    
    print(f"Current annotation stats:")
    print(f"  - Label values: {unique_values}")
    print(f"  - Labeled pixels: {labeled_pixels}")
    print(f"  - Total size: {labels_data.shape}")
    
    # 如果为空，提示用户
    if labeled_pixels == 0:
        print("No annotations in current view.")
    else:
        # 创建一个临时的标注查看层，更明显地显示当前标注
        temp_view = np.zeros(labels_data.shape, dtype=np.uint8)
        temp_view[labels_data > 0] = 255
        # 移除之前的debug视图（如果有）
        for layer in list(viewer.layers):
            if layer.name.startswith("Current Annotations Debug"):
                viewer.layers.remove(layer)
        # 添加新的debug视图
        viewer.add_image(temp_view, name="Current Annotations Debug", colormap='yellow')

# Create custom button widgets
control_widget = QWidget()
layout = QVBoxLayout()
control_widget.setLayout(layout)

# Next image button
next_button = QPushButton("Next Image")
next_button.clicked.connect(load_next_image)
layout.addWidget(next_button)

# Save annotation button
save_button = QPushButton("Save Current Annotation")
save_button.clicked.connect(save_current_annotations)
layout.addWidget(save_button)

# View annotation button
view_button = QPushButton("View Current Annotation")
view_button.clicked.connect(view_current_annotations)
layout.addWidget(view_button)

# 添加刷新界面按钮
def refresh_view():
    global labels_layer, tile_layer
    # 清理临时可视化层
    for layer in list(viewer.layers):
        if layer.name.startswith("Current Annotations Debug") or layer.name.startswith("Debug: mask_"):
            viewer.layers.remove(layer)
    
    # 强制重新显示当前图层
    temp_data = tile_layer.data.copy()
    tile_layer.data = temp_data
    
    # 强制重新显示标注层
    temp_labels = labels_layer.data.copy()
    labels_layer.data = np.zeros_like(temp_labels)  # 先清空
    labels_layer.data = temp_labels  # 再恢复
    
    print("Interface has been refreshed")

refresh_button = QPushButton("Refresh Interface")
refresh_button.clicked.connect(refresh_view)
layout.addWidget(refresh_button)

# Train model button
train_button = QPushButton("Train Model")
train_button.clicked.connect(train_model)
layout.addWidget(train_button)

# Add widget to napari
viewer.window.add_dock_widget(control_widget, area="right", name="Controls")

# Initialize current annotation object
current_annotation = {
    "tile_position": (int(x_offset), int(y_offset)),
    "tile_size": TILE_SIZE,
    "has_label_mask": False,
    "timestamp": str(np.datetime64('now'))
}

# Display usage instructions
print("""
Annotation Instructions:
1. Use the paint brush tool (brush icon) in the left toolbar to draw contours on the "Contour Annotation" layer
2. After drawing contours on the current image, click "Save Current Annotation" button on the right panel
3. Click "Next Image" button to load a new image
4. After annotating all images, click "Train Model" button to start model training
5. Click "View Current Annotation" to check if your annotations are being captured

Note: 
- The annotation layer now uses a bright yellow color for better visibility
- You must click "Save Current Annotation" button after drawing to save your work
- If you don't see your annotations, try using the "View Current Annotation" button
""")

# Run the program
napari.run()