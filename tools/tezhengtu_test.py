
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
import numpy as np
import matplotlib.pyplot as plt
import mmcv

def visualize_feature_map(model, img_path, layer_name, device='cuda'):
    # Load image
    img = mmcv.imread(img_path)

    # Initialize segmentor
    model = init_segmentor(model, device=device)

    # Perform inference
    result = inference_segmentor(model, img)

    # Get feature map
    feature_map = model.forward_dummy(img.unsqueeze(0).to(device), return_loss=False, return_feats=True)

    # Get the activations from the desired layer
    activations = feature_map[layer_name].detach().cpu().numpy()

    # Number of features in the feature map
    num_features = activations.shape[1]

    # Size of the grid for visualization (e.g., 8x8 if 64 features)
    grid_size = int(np.ceil(np.sqrt(num_features)))

    # Visualization
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            if i * grid_size + j < num_features:
                ax[i][j].imshow(activations[0, i * grid_size + j], cmap='viridis')
                ax[i][j].axis('off')
    plt.show()

# Path to the model config file
config_file = r'work_dir_2024_0304\segformer_band3_b2\segformer_mit-b2_512x512_160k_mywater.py'
# Path to the checkpoint file
checkpoint_file = r'work_dir_2024_0304\segformer_band3_b2\iter_160000.pth'

# Load model
model = init_segmentor(config_file, checkpoint_file)

# Visualize feature map of a specific layer
layer_name = 'feature_map2_0'  # Change this to the desired layer name
img_path = 'H48F020017_clip4_70_0_0.png'  # Provide the path to your image

# Visualize feature map
visualize_feature_map(model, img_path, layer_name)
