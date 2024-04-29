import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

names = {
    'best_model_i': 'Metrics for best model',
    'flow_only_i': 'Metrics for flow-only predictions',
    'no_feature_map_i': 'Metrics for model with no feature map encoder',
    'flow_roi_est_i': 'Metrics model with flow-guided training',
    'small_rand_roi_i': 'Metrics for model with small random ROI growth',
    'mse_loss_i': 'Metrics for model with mean squared error loss',
}

for csv_file in glob.glob('metrics/*.csv'):
    csv_file_path = Path(csv_file)
    stem = csv_file_path.stem

    if stem not in names:
        continue

    df = pd.read_csv(csv_file)

    plt.figure(figsize=(5, 4))
    plt.subplots_adjust(left=0.17, right=0.95)

    metric_columns = [col for col in df.columns if col != 'Unnamed: 0']
    for metric_column in metric_columns:
        plt.plot(df.iloc[:,0], df[metric_column], label=metric_column)

    plt.title(names[stem])
    plt.xlabel('Frame Number')
    plt.ylabel('Metric Value')
    plt.ylim(-0.5, 1.0)
    
    plt.legend(loc='lower left')
    
    plot_file_path = csv_file_path.parent / f"{stem}_plot.png"
    plt.savefig(plot_file_path)
    plt.close()
