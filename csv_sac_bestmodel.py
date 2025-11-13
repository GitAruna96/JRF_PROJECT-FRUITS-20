import torch
import pandas as pd
import numpy as np
import random
from timm import create_model
import torch.optim as optim
import torch.serialization

# Add safe globals to allow numpy objects
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

# Load the complete checkpoint without weights_only restriction
checkpoint_path = r'C:\\Users\\Admin\\Aruna_jrf_projects\Datasets\\fruits_dataset\\paper_experiments\\results\\checkpoint_epoch_17_78.45.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("✓ SUCCESS: Loaded complete checkpoint!")
print("=" * 50)

# Extract all information
print(f"Epoch: {checkpoint['epoch']}")
print(f"Test Accuracy: {checkpoint['test_accuracy']:.2f}%")
print(f"Confidence Threshold: {checkpoint['conf_threshold']:.3f}")

# Extract the complete training history
history = checkpoint['history']
print(f"Training history contains {len(history['train_cls_loss'])} epochs")

# Recreate the CSV with all data
df = pd.DataFrame({
    'Epoch': list(range(1, len(history['train_cls_loss']) + 1)),
    'Train_CLS_Loss': history['train_cls_loss'],
    'Train_SAC_Loss': history['train_sac_loss'],
    'Test_Accuracy': history['test_acc'],
    'Test_Loss': history['test_loss'],
    'Conf_Threshold': history['conf_threshold'],
    'Epoch_Time_s': history['epoch_time'],
    'Cumulative_Time_s': history['cumulative_time']
})

# Save the recovered CSV
recovered_csv_path = './recovered_best_results_78.45.csv'
df.to_csv(recovered_csv_path, index=False)
print(f"✓ Recovered results saved to: {recovered_csv_path}")

# Find the best epoch and accuracy
best_epoch = np.argmax(history['test_acc']) + 1
best_accuracy = max(history['test_acc'])
print(f"Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")

# Show the exact parameters that worked
print("\n" + "=" * 50)
print("🎯 SUCCESSFUL PARAMETERS THAT GAVE 78.45%:")
print("=" * 50)
print(f"Best Confidence Threshold: {checkpoint['conf_threshold']:.3f}")
print(f"Final Model Accuracy: {checkpoint['test_accuracy']:.2f}%")
print(f"Total Training Epochs: {len(history['train_cls_loss'])}")
print(f"Best Achieved Accuracy: {best_accuracy:.2f}%")

# Restore the exact model and optimizer
model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=20)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print("✓ Model and optimizer states restored!")

# Restore random states for exact reproducibility
random.setstate(checkpoint['random_state']['python'])
np.random.set_state(checkpoint['random_state']['numpy'])
torch.set_rng_state(checkpoint['random_state']['torch'])

print("✓ Random states restored for exact reproducibility!")

# Show training progression
print("\n" + "=" * 50)
print("📊 TRAINING PROGRESSION SUMMARY:")
print("=" * 50)
print(f"Initial Accuracy: {history['test_acc'][0]:.1f}%")
print(f"Final Accuracy: {history['test_acc'][-1]:.1f}%")
print(f"Accuracy Improvement: {history['test_acc'][-1] - history['test_acc'][0]:.1f}%")
print(f"Final CLS Loss: {history['train_cls_loss'][-1]:.4f}")
print(f"Final SAC Loss: {history['train_sac_loss'][-1]:.4f}")

# Compare with baseline
baseline = 78.33
improvement = best_accuracy - baseline
print(f"\nComparison with Baseline ({baseline}%):")
print(f"Your Improvement: {improvement:+.2f}%")
if improvement > 0:
    print("🎉 YOU BEAT THE BASELINE!")
else:
    print("⚠️  Close to baseline - let's optimize further!")