"""
Multi-Model Image Classification Comparison using PyTorch
===========================================================
This script compares multiple pretrained models for flower classification:
- ResNet50
- VGG16
- EfficientNet-B0
- MobileNetV3
- ConvNeXt-Tiny
- DenseNet121

Author: Machine Learning Assignment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 5  # Number of epochs per model
LEARNING_RATE = 0.001
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# Data Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# Load Datasets
# ============================================================
print("\n" + "="*60)
print("Loading Datasets...")
print("="*60)

train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
val_dataset = datasets.ImageFolder("dataset/valid", transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Flower names mapping (Oxford Flowers 102)
flower_names = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
    'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle',
    'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris',
    'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily',
    'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth',
    'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william',
    'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
    'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose',
    'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower',
    'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy',
    'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium',
    'orange dahlia', 'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan',
    'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris',
    'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose',
    'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily',
    'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine',
    'desert-rose', 'tree mallow', 'magnolia', 'cyclamen', 'watercress',
    'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove',
    'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia',
    'blanket flower', 'trumpet creeper', 'blackberry lily'
]

# ============================================================
# Model Definitions
# ============================================================
def create_model(model_name, num_classes, pretrained=True):
    """
    Create a pretrained model with a custom classification head.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        model: PyTorch model ready for training
    """
    weights = 'IMAGENET1K_V1' if pretrained else None
    
    if model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        # Freeze base layers
        for param in model.parameters():
            param.requires_grad = False
        # Replace classifier
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    elif model_name == 'vgg16':
        model = models.vgg16(weights=weights)
        for param in model.features.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_large(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(DEVICE)

# ============================================================
# Training Function
# ============================================================
def train_model(model, model_name, train_loader, val_loader, epochs=5):
    """
    Train a model and return training history.
    """
    print(f"\n{'='*60}")
    print(f"🚀 Training: {model_name.upper()}")
    print(f"{'='*60}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', ncols=100)
        for images, labels in train_pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total:.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]  ', ncols=100)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{val_correct/val_total:.4f}'})
        
        scheduler.step()
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['epoch_time'].append(epoch_time)
        
        # Update best accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
        
        print(f"\n📊 Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s\n")
    
    history['best_val_acc'] = best_val_acc
    return history

# ============================================================
# Main: Train All Models
# ============================================================
# List of models to compare
MODEL_LIST = [
    'resnet50',
    'vgg16', 
    'efficientnet_b0',
    'mobilenet_v3',
    'convnext_tiny',
    'densenet121'
]

# Store all results
all_results = {}

print("\n" + "="*60)
print("🔬 MULTI-MODEL COMPARISON EXPERIMENT")
print("="*60)
print(f"Models to compare: {MODEL_LIST}")
print(f"Epochs per model: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print("="*60)

for model_name in MODEL_LIST:
    try:
        # Create model
        model = create_model(model_name, num_classes)
        
        # Train model
        history = train_model(model, model_name, train_loader, val_loader, epochs=EPOCHS)
        
        # Store results
        all_results[model_name] = history
        
        # Clean up to free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"❌ Error training {model_name}: {e}")
        continue

# ============================================================
# Results Summary
# ============================================================
print("\n" + "="*60)
print("📊 FINAL RESULTS SUMMARY")
print("="*60)

# Create comparison table
print(f"\n{'Model':<20} {'Best Val Acc':<15} {'Final Val Acc':<15} {'Total Time':<15}")
print("-" * 65)

best_model = None
best_acc = 0

for model_name, history in all_results.items():
    best_val_acc = history['best_val_acc']
    final_val_acc = history['val_acc'][-1]
    total_time = sum(history['epoch_time'])
    
    print(f"{model_name:<20} {best_val_acc*100:.2f}%{'':<8} {final_val_acc*100:.2f}%{'':<8} {total_time:.1f}s")
    
    if best_val_acc > best_acc:
        best_acc = best_val_acc
        best_model = model_name

print("-" * 65)
print(f"\n🏆 BEST MODEL: {best_model.upper()} with {best_acc*100:.2f}% validation accuracy!")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Training Accuracy
ax1 = axes[0, 0]
for model_name, history in all_results.items():
    ax1.plot(range(1, EPOCHS+1), history['train_acc'], marker='o', label=model_name)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Accuracy')
ax1.set_title('Training Accuracy Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Validation Accuracy
ax2 = axes[0, 1]
for model_name, history in all_results.items():
    ax2.plot(range(1, EPOCHS+1), history['val_acc'], marker='s', label=model_name)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Accuracy')
ax2.set_title('Validation Accuracy Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Training Loss
ax3 = axes[1, 0]
for model_name, history in all_results.items():
    ax3.plot(range(1, EPOCHS+1), history['train_loss'], marker='o', label=model_name)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
ax3.set_title('Training Loss Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Best Accuracy Bar Chart
ax4 = axes[1, 1]
models = list(all_results.keys())
accuracies = [all_results[m]['best_val_acc'] * 100 for m in models]
colors = ['#2ecc71' if m == best_model else '#3498db' for m in models]
bars = ax4.bar(models, accuracies, color=colors)
ax4.set_xlabel('Model')
ax4.set_ylabel('Best Validation Accuracy (%)')
ax4.set_title('Best Accuracy by Model')
ax4.set_ylim([min(accuracies)-5, 100])
for bar, acc in zip(bars, accuracies):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Results saved to 'model_comparison_results.png'")
print("="*60)
