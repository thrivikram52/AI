import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available - GPU makes training much faster!
# For Mac: prioritize MPS (Apple Silicon GPU) over CUDA
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device} (Apple Metal GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device} (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print(f"Using device: {device} (CPU only)")

# ===================================================================
# 1. LOAD DATA WITH BUILT-IN TRANSFORMS
# ===================================================================
print("Loading MNIST dataset...")

# Transform pipeline: PIL Image → PyTorch Tensor → Normalized values
transform = transforms.Compose([
    transforms.ToTensor(),                    # Converts PIL image to tensor and scales to [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # Centers data around 0 for better training
])

# Download MNIST dataset (60k training + 10k test images)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders - handles batching and shuffling automatically
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)    # 64 images per batch, randomized
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)   # Larger batches for testing (no need to shuffle)

print(f"Training samples: {len(train_dataset)}")  # Should print 60,000
print(f"Test samples: {len(test_dataset)}")       # Should print 10,000

# ===================================================================
# 2. VISUALIZE SOME DATA - See what we're working with!
# ===================================================================
def show_samples():
    # Get one batch of training data (64 images)
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Plot first 8 images in a 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(8):
        row, col = i // 4, i % 4  # Calculate grid position
        # Remove normalization for proper display (reverse the transform)
        img = images[i].squeeze() * 0.3081 + 0.1307  # Undo: (pixel - 0.1307) / 0.3081
        axes[row, col].imshow(img, cmap='gray')       # Display as grayscale image
        axes[row, col].set_title(f'Label: {labels[i].item()}')  # Show true digit
        axes[row, col].axis('off')  # Remove axis numbers for cleaner look
    
    plt.suptitle('Sample MNIST Digits')
    plt.tight_layout()
    plt.show()

show_samples()  # Execute the visualization

# ===================================================================
# 3. DEFINE THE NEURAL NETWORK - The Brain of Our AI!
# ===================================================================
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # PyTorch automatically initializes weights using smart defaults!
        self.flatten = nn.Flatten()              # Converts 2D image (28x28) to 1D vector (784)
        
        # Sequential container: data flows through layers in order
        self.layers = nn.Sequential(
            nn.Linear(784, 128),                 # First layer: 784 inputs → 128 neurons
            nn.ReLU(),                           # Activation: outputs max(0, x) - adds non-linearity
            nn.Dropout(0.2),                     # Regularization: randomly turn off 20% of neurons during training
            nn.Linear(128, 64),                  # Second layer: 128 → 64 neurons  
            nn.ReLU(),                           # Another activation function
            nn.Linear(64, 10)                    # Output layer: 64 → 10 (one for each digit 0-9)
        )
    
    def forward(self, x):
        x = self.flatten(x)                      # Convert image to vector: (28,28) → (784,)
        return self.layers(x)                    # Pass through all layers and return final predictions

# Create the model and move it to GPU/CPU
model = SimpleMLP().to(device)
print("\nModel architecture:")
print(model)

# Count total learnable parameters - these will be adjusted during training
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")  # Should be around 101,770

# ===================================================================
# 4. SETUP TRAINING COMPONENTS
# ===================================================================
criterion = nn.CrossEntropyLoss()               # Loss function: measures how wrong our predictions are
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Smart optimizer: adjusts weights to minimize loss

# ===================================================================
# 5. TRAINING FUNCTION - Where the Magic Happens!
# ===================================================================
def train_model(epochs=10):
    model.train()  # Enable training mode (activates dropout, batch norm, etc.)
    train_losses = []  # Track loss over time to see learning progress
    
    # One epoch = showing the model all 60,000 training images once
    for epoch in range(epochs):
        running_loss = 0.0  # Accumulate loss for this epoch
        correct = 0         # Count correct predictions
        total = 0           # Count total predictions
        
        # Process data in batches of 64 images each
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move current batch to GPU/CPU
            data, target = data.to(device), target.to(device)
            
            # THE CORE TRAINING STEPS (this is where learning happens!)
            optimizer.zero_grad()               # Clear leftover gradients from previous batch
            output = model(data)                # Forward pass: get model predictions
            loss = criterion(output, target)    # Calculate how wrong we are
            loss.backward()                     # Backward pass: calculate gradients (how to improve)
            optimizer.step()                    # Update model weights based on gradients
            
            # Track statistics for monitoring progress
            running_loss += loss.item()                           # Accumulate loss
            _, predicted = torch.max(output.data, 1)              # Get predicted class (highest probability)
            total += target.size(0)                               # Count images in this batch
            correct += (predicted == target).sum().item()         # Count correct predictions
            
            # Print progress every 200 batches so we know training is working
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate statistics for this complete epoch
        epoch_loss = running_loss / len(train_loader)  # Average loss per batch
        epoch_acc = 100 * correct / total              # Accuracy percentage
        train_losses.append(epoch_loss)                # Save for plotting later
        
        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, '
              f'Accuracy: {epoch_acc:.2f}%')
    
    return train_losses

# ===================================================================
# 6. TESTING FUNCTION - Evaluate on Unseen Data
# ===================================================================
def test_model():
    model.eval()  # Switch to evaluation mode (disables dropout, etc.)
    test_loss = 0  # Accumulate test loss
    correct = 0    # Count correct predictions
    
    # No gradients needed for testing - saves memory and speeds up computation
    with torch.no_grad():
        # Process all test data in batches
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)                                    # Get predictions
            test_loss += criterion(output, target).item()           # Accumulate loss
            pred = output.argmax(dim=1, keepdim=True)               # Get predicted class
            correct += pred.eq(target.view_as(pred)).sum().item()   # Count correct predictions
    
    # Calculate final test statistics
    test_loss /= len(test_loader)                    # Average loss
    accuracy = 100. * correct / len(test_dataset)    # Percentage accuracy
    
    print(f'\nTest Results:')
    print(f'Average loss: {test_loss:.4f}')
    print(f'Accuracy: {correct}/{len(test_dataset)} ({accuracy:.2f}%)')
    
    return accuracy

# ===================================================================
# 7. PREDICTION FUNCTION - See the Model in Action!
# ===================================================================
def predict_sample():
    model.eval()  # Set to evaluation mode
    
    # Get one batch of test images (1000 images)
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Take first 6 images for visualization
    sample_images = images[:6].to(device)  # Move to GPU/CPU for prediction
    sample_labels = labels[:6]             # Keep labels on CPU for display
    
    # Make predictions without computing gradients
    with torch.no_grad():
        outputs = model(sample_images)              # Get raw outputs (probabilities)
        _, predictions = torch.max(outputs, 1)     # Convert to predicted digit (0-9)
    
    # Display results in a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for i in range(6):
        row, col = i // 3, i % 3  # Calculate grid position
        
        # Prepare image for display (reverse normalization and move to CPU)
        img = sample_images[i].cpu().squeeze() * 0.3081 + 0.1307
        axes[row, col].imshow(img, cmap='gray')
        
        # Show both true label and model prediction
        axes[row, col].set_title(f'True: {sample_labels[i].item()}, '
                                f'Pred: {predictions[i].item()}')
        axes[row, col].axis('off')
    
    plt.suptitle('Model Predictions')
    plt.tight_layout()
    plt.show()

# ===================================================================
# 8. RUN EVERYTHING! - Execute the Complete Pipeline
# ===================================================================
print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50)

# Train the model for 10 epochs (10 complete passes through the data)
train_losses = train_model(epochs=10)

print("\n" + "="*50)
print("TESTING MODEL")
print("="*50)

# Evaluate the trained model on test data
final_accuracy = test_model()

# Show some predictions to see the model in action
predict_sample()

# Create visualizations of training progress and final results
plt.figure(figsize=(10, 4))

# Plot 1: Training loss over time (should decrease)
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot 2: Final test accuracy (should be high)
plt.subplot(1, 2, 2)
plt.bar(['Final Accuracy'], [final_accuracy])
plt.title('Test Accuracy')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

print(f"\n🎉 Training complete! Final accuracy: {final_accuracy:.2f}%")