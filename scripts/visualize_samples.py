# scripts/visualize_samples.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

def clean_token(token):
    """
    Cleans subword tokens (like DeBERTa's ' ' or 'Ġ' or '▁') 
    to reconstruct readable text.
    """
    t = str(token)
    if t.startswith('▁') or t.startswith(' '):
        return ' ' + t[1:]
    if t.startswith('##'):
        return t[2:]
    return t

def wrap_text_with_colors(ax, tokens, colors, x_start, y_start, max_width, fontsize=11):
    """
    Draws text wrapped at max_width. Each token can have a specific color.
    """
    font = FontProperties(family='sans-serif', size=fontsize)
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    
    x, y = x_start, y_start
    line_height = 0.03  
    
    for token, color in zip(tokens, colors):
        t = ax.text(x, y, token, color=color, fontproperties=font, 
                    transform=ax.transAxes, verticalalignment='top')
        
        bbox = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        width = bbox.width
        
        if x + width > x_start + max_width:
            x = x_start
            y -= line_height
            t.set_position((x, y))
            bbox = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
            width = bbox.width
            
        x += width
        
    return y - line_height

def plot_document(df_doc, doc_id, ax, max_width=0.96):
    """
    Plots the three versions of a single document on a given matplotlib axis.
    """
    tokens = [clean_token(t) for t in df_doc['token'].tolist()]
    true_labels = df_doc['true_label'].tolist()
    pred_labels = df_doc['pred_label'].tolist()
    
    COLOR_HUMAN = '#2ca02c'   # Verde
    COLOR_AI = '#1f77b4'      # Blu
    COLOR_CORRECT = '#000000' # Nero
    COLOR_MISTAKE = '#d62728' # Rosso
    
    colors_true = [COLOR_AI if l == 1 else COLOR_HUMAN for l in true_labels]
    colors_pred = [COLOR_AI if l == 1 else COLOR_HUMAN for l in pred_labels]
    colors_mistake = [COLOR_MISTAKE if t != p else COLOR_CORRECT for t, p in zip(true_labels, pred_labels)]

    y_pos = 0.98
    x_pos = 0.02
    
    ax.text(x_pos, y_pos, f"Document ID: {doc_id}", fontsize=14, fontweight='bold', transform=ax.transAxes)
    y_pos -= 0.04
    
    ax.text(x_pos, y_pos, "1) Ground Truth (Green = Human, Blue = AI):", fontsize=12, fontweight='bold', transform=ax.transAxes)
    y_pos -= 0.03
    y_pos = wrap_text_with_colors(ax, tokens, colors_true, x_pos, y_pos, max_width)
    y_pos -= 0.06
    
    ax.text(x_pos, y_pos, "2) Predicted Labels (Green = Human, Blue = AI):", fontsize=12, fontweight='bold', transform=ax.transAxes)
    y_pos -= 0.03
    y_pos = wrap_text_with_colors(ax, tokens, colors_pred, x_pos, y_pos, max_width)
    y_pos -= 0.06
    
    ax.text(x_pos, y_pos, "3) Mistakes (Red = Error, Black = Correct):", fontsize=12, fontweight='bold', transform=ax.transAxes)
    y_pos -= 0.03
    y_pos = wrap_text_with_colors(ax, tokens, colors_mistake, x_pos, y_pos, max_width)

    ax.set_axis_off()

def generate_visualization(csv_path, output_png, num_samples=5):
    print(f"📊 Reading {csv_path} for visualization...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: File {csv_path} not found.")
        return False

    unique_docs = df['doc_id'].unique()[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 12 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    fig.suptitle("Token-Level Predictions Visualization", fontsize=20, fontweight='bold', y=0.99)

    for i, doc_id in enumerate(unique_docs):
        print(f"   Processing Doc ID {doc_id} for image...")
        df_doc = df[df['doc_id'] == doc_id]
        plot_document(df_doc, doc_id, axes[i])

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved visualization to {output_png}")
    return True