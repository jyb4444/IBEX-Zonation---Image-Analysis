import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from skimage.transform import resize
import imageio

# Example U-Net Model (replace with your actual U-Net)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """
        U-Net architecture.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB, 1 for grayscale).
            out_channels (int): Number of output channels (e.g., 1 for binary segmentation).
            features (list): List of feature map sizes for each layer.
        """
        super(UNet, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder_blocks.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoder_blocks.append(UpConv(feature * 2, feature))
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))

        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse skip connections

        for decoder_up, decoder_conv in zip(self.decoder_blocks[0::2], self.decoder_blocks[1::2]):
            x = decoder_up(x)
            skip_connection = skip_connections.pop()
            x = torch.cat((skip_connection, x), dim=1)
            x = decoder_conv(x)

        return self.final_conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

def reconstruct_full_mask(annotations, image_shape):
    """Reconstructs the full label mask from tiled masks."""
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for annotation in annotations:
        if annotation["has_label_mask"]:
            mask_data = np.load(annotation["mask_file"])
            x_offset, y_offset = annotation["tile_position"]
            tile_size = annotation["tile_size"]
            full_mask[x_offset:x_offset + tile_size, y_offset:y_offset + tile_size] = mask_data
    return full_mask

def prepare_training_data(image, full_mask):
    """Prepares training data from the full image and mask, with channel-wise normalization."""
    num_channels = image.shape[2]
    normalized_channels = []

    for channel_idx in range(num_channels):
        channel = image[:, :, channel_idx]
        channel = (channel - channel.min()) / (channel.max() - channel.min())
        normalized_channels.append(channel)

    normalized_image = np.stack(normalized_channels, axis=2)
    image_tensor = torch.tensor(normalized_image.transpose((2, 0, 1)), dtype=torch.float32) # Channels first
    mask_tensor = torch.tensor(full_mask[np.newaxis, :, :], dtype=torch.float32)

    return image_tensor, mask_tensor

def calculate_iou(prediction, target):
    """Calculates the Intersection over Union (IoU)."""
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou

def export_labeled_image(image, full_mask, output_path):
    """Exports a PNG image with labeled regions overlaid."""
    image = np.mean(image, axis=2) # Convert to grayscale
    image = (image - image.min()) / (image.max() - image.min()) * 255 # Normalize to 0-255

    labeled_image = np.stack([image, image, image], axis=-1).astype(np.uint8) # Create RGB image

    # Add red overlay for labeled regions
    labeled_image[full_mask > 0, 0] = 255
    labeled_image[full_mask > 0, 1] = 0
    labeled_image[full_mask > 0, 2] = 0

    output_filename = os.path.join(output_path, "labeled_image.png")
    imageio.imwrite(output_filename, labeled_image)
    print(f"Labeled image exported to: {output_filename}")

def train_model(annotations, image, viewer, output_path, model=None, optimizer=None):
    """Trains the U-Net model using annotations and the full image."""
    if not annotations:
        print("No annotations found. Cannot train model.")
        return

    full_mask = reconstruct_full_mask(annotations, image.shape)
    image_tensor, mask_tensor = prepare_training_data(image, full_mask)

    # Export labeled image
    export_labeled_image(image, full_mask, output_path)

    if model is None:
        # Update the input channel count to match the number of channels in the image
        model = UNet(in_channels=image.shape[2]) # Replace with your U-Net, and pass the correct number of channels.
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 10
    batch_size = 1

    dataset = TensorDataset(image_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)) # Add batch dimension
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_iou = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Calculate IoU
            predictions = torch.sigmoid(outputs).detach().numpy() > 0.5
            targets_np = targets.detach().numpy()
            epoch_iou += calculate_iou(predictions, targets_np)

        avg_loss = epoch_loss / len(dataloader)
        avg_iou = epoch_iou / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}')

    print("Model training completed.")

    # Save the model's state_dict
    model_save_path = os.path.join(output_path, "trained_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    apply_model_to_image(model, image, viewer)

def apply_model_to_image(model, image, viewer):
    """Applies the trained model to the full image and displays the prediction."""
    print("Applying model to the entire image...")
    try:
        model.eval()
        model.to('cpu') # Ensure model is on CPU for inference.
        num_channels = image.shape[2]
        normalized_channels = []

        for channel_idx in range(num_channels):
            channel = image[:, :, channel_idx]
            channel = (channel - channel.min()) / (channel.max() - channel.min())
            normalized_channels.append(channel)

        normalized_image = np.stack(normalized_channels, axis=2)
        image_tensor = torch.tensor(normalized_image.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0) # Add batch dimension

        chunk_size = 1024
        height, width = image.shape[0], image.shape[1]
        prediction = np.zeros((height, width), dtype=np.float32)

        with torch.no_grad():
            for y in range(0, height, chunk_size):
                for x in range(0, width, chunk_size):
                    y_end = min(y + chunk_size, height)
                    x_end = min(x + chunk_size, width)
                    chunk = image_tensor[:, :, :, y:y_end, x:x_end]
                    chunk_pred = torch.sigmoid(model(chunk)).squeeze().numpy() # Apply sigmoid and convert to numpy
                    prediction[y:y_end, x:x_end] = chunk_pred

            viewer.add_image((prediction > 0.5).astype(float), name="Model Prediction")
            print("Prediction completed and displayed")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")