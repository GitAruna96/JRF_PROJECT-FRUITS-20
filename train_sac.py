import os
import torch
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft
from data_load_sac import get_loader
from timm import create_model
import time

# Spectral Perturbation (from previous response)
def spectral_perturbation(x, noise_scale = 0.05, high_freq_ratio = 0.7):
    freq = fft.fft2(x)
    freq_shift = fft.fftshift(freq)
    h, w = x.shape[-2:]
    cy, cx = h //2, w //2
    radius = int(min(h,w) * high_freq_ratio/2)
    mask = torch.ones((h,w), device = x.device)
    mask[cy-radius:cy+radius, cx-radius:cx+radius] = 0
    noise = noise_scale * torch.randn_like(freq_shift) * mask
    freq_perturbed = freq_shift +noise
    freq_back = fft.ifftshift(freq_perturbed)
    x_perturbed = fft.ifft2(freq_back).real.clamp(0,1)
    return x_perturbed

# Training Loop for SAC Anchor

def train_sac(model,src_train_loader,trg_train_loader,optimizer,criterion_cls,criterion_mse,conf_threshold,lambda_sac,device,epochs):
    
    for epoch in range(epochs):
        model.train()
        total_cls_loss = 0.0
        total_sac_loss = 0.0
        num_batches = 0

        for src_batch, trg_batch in zip(src_train_loader, trg_train_loader):
            src_imgs, src_labels = src_batch[0].to(device), src_batch[1].to(device)
            trg_imgs = trg_batch[0].to(device)

            #source classificaton
            src_feats = model(src_imgs)
            cls_loss = criterion_cls(src_feats, src_labels)
            total_cls_loss +=cls_loss.item()

            #target: sac anchor

            trg_feats = model(trg_imgs)
            trg_probs = torch.softmax(trg_feats, dim=1)
            conf_scores, _ = torch.max(trg_probs, dim=1)
            anchor_mask = conf_scores > conf_threshold
            anchor_imgs = trg_imgs[anchor_mask]
            anchor_feats = trg_feats[anchor_mask]

            sac_loss = torch.tensor(0.0, device=device)
            if anchor_imgs.size(0) > 0:
                perturbed_anchors = spectral_perturbation(anchor_imgs)
                perturbed_feats = model(perturbed_anchors)
                sac_loss = criterion_mse(anchor_feats, perturbed_feats)
                total_sac_loss +=sac_loss.item()

            #total loss
            total_loss = cls_loss+lambda_sac * sac_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            num_batches +=1

        avg_cls_loss = total_cls_loss / num_batches
        avg_sac_loss = total_sac_loss / num_batches

    return avg_cls_loss, avg_sac_loss
        #print(f"Epoch {epoch+1}/{epochs} - CLS Loss: {avg_cls_loss:.4f}, SAC Loss: {avg_sac_loss:.4f}")

        # Evaluate (add your accuracy code here)
def evaluate(model, trg_evaluate_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in trg_evaluate_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

    return accuracy
def save_history_to_csv(history, model_name, output_dir):
    epochs = list(range(1, len(history['train_cls_loss']) + 1))
    df = pd.DataFrame({
        'Epoch': epochs,
        'Train_Loss': history['train_cls_loss'],
        'Train_Accuracy': history['train_acc'],
        'Test_Loss': history['test_loss'],
        'Test_Accuracy': history['test_acc'],
        'Epoch_Time_s': history['epoch_time'],  
        'Cumulative_Time_s': history['cumulative_time']  
    })
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'{model_name}_fruits_sac_results_{timestamp}.csv')
    
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    return csv_path

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"python executable: {sys.executable}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_names = ['swin_t']
    batch_size = 32
    no_epochs = 50
    lr = 1e-4
    lambda_sac = 1.0
    conf_threshold = 0.85
    output_classes = 20
    patience=100
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    # Loaders (using corrected get_loader)
    src_train_loader = get_loader('fruitsO', batch_size, train=True, domain='source')
    trg_train_loader = get_loader('fruitsP', batch_size, train=True, domain='target')
    trg_evaluate_loader = get_loader('fruitsP', batch_size, train=False, domain='target') 

    no_classes = len(src_train_loader.dataset.classes)
    print(f"Number of classes: {no_classes}")
    assert no_classes == output_classes, f"Expected {output_classes} classes found {no_classes}"

    histories = {} 
    for model_name in model_names: 
        print(f"\nTraining {model_name} model...") 
        model = create_model('swin_tiny_patch4_window7_224', pretrained = True, num_classes = num_classes)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr = lr)

        criterion_cls  = nn.CrossEntropyLoss()
        criterion_mse = nn.MSELoss()

        history = {'train_cls_loss': [], 'train_acc': [], 'test_acc': [], 'test_loss': [],'sac_loss': [], 'epoch_time': [], 'cumulative_time': []}
        counter = 0
        best_acc = 0.0

        cumulative_time = 0

        for epoch in range(no_epochs):

            train_cls_loss, train_acc, epoch_time = train_sac(model,src_train_loader,trg_train_loader,optimizer,criterion_cls,criterion_mse,conf_threshold,lambda_sac,device,no_epochs)
            test_loss, test_acc = evaluate(model, trg_evaluate_loader, device)
            cumulative_time += epoch_time

            history['train_cls_loss'].append(train_cls_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['test_loss'].append(test_loss)
            history['epoch_time'].append(epoch_time)
            history['cumulative_time'].append(cumulative_time)
            
            #scheduler.step()

            print(f"{model_name}-Epoch {epoch+1}/{no_epochs},"
                f"Time: {epoch_time:.2f}s (Total: {cumulative_time:.2f}s), "
                f"Train Loss: {train_cls_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                counter = 0
                torch.save(model.state_dict(), f'fruits_sac_{model_name}.pth')
                print(f"{model_name} - New best accuracy: {best_acc:.2f}%. Model saved.")
            else:
                counter += 1
                print(f"{model_name} - No improvement. Patience counter: {counter}/{patience}")
            
            if counter >= patience:
                print(f"{model_name} - Early stopping triggered after {patience} epochs with no improvement.")
                break

        # FINAL TIMING ANALYSIS R
        print(f"\n--- {model_name} Timing Analysis ---")
        print(f"Total Training Time: {cumulative_time:.2f} seconds ({cumulative_time/60:.2f} minutes)")
        print(f"Average Time per Epoch: {np.mean(history['epoch_time']):.2f} seconds")
        print(f"Fastest Epoch: {np.min(history['epoch_time']):.2f} seconds")
        print(f"Slowest Epoch: {np.max(history['epoch_time']):.2f} seconds")

        histories[model_name] = history
        save_history_to_csv(history, model_name, output_dir)
        print(f'{model_name} - Training finished. Best Test Accuracy: {best_acc:.2f}%')

    print("\n Final comparision")
    for model_name in model_names:
        best_acc = max(histories[model_name]['test_acc'])
        print(f"{model_name} - Best Test Accuracy: {best_acc:.2f}%")
    return histories

if __name__ == '__main__':
    histories = main()

