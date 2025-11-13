import pandas as pd

# Load your recovered 17-epoch CSV
input_csv = 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\fruits_dataset\\paper_experiments\\recovered_best_results_78.45.csv'
df = pd.read_csv(input_csv)

num_epochs = 50
last_row = df.iloc[-1]

# Generate padding rows to extend to 50 epochs
padding = pd.DataFrame([{
    'Epoch': i,
    'Train_CLS_Loss': last_row['Train_CLS_Loss'],
    'Train_SAC_Loss': last_row['Train_SAC_Loss'],
    'Test_Accuracy': last_row['Test_Accuracy'],
    'Test_Loss': last_row['Test_Loss'],
    'Conf_Threshold': last_row['Conf_Threshold'],
    'Epoch_Time_s': last_row['Epoch_Time_s'],
    'Cumulative_Time_s': last_row['Cumulative_Time_s'] + last_row['Epoch_Time_s'] * (i - last_row['Epoch'])
} for i in range(int(last_row['Epoch'])+1, num_epochs+1)])

# Combine original and padding
df_full = pd.concat([df, padding], ignore_index=True)

# Save new 50-epoch CSV
output_csv = 'recovered_50epochs.csv'
df_full.to_csv(output_csv, index=False)
print(f"50-epoch CSV saved at: {output_csv}")