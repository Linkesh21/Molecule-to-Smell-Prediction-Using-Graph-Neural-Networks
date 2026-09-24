import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_flowchart(output_path="methodology_flowchart.png"):
    # High resolution 16:10 aspect ratio with ample vertical margins
    fig, ax = plt.subplots(figsize=(19, 13.5), dpi=300)
    ax.set_facecolor('#0B1120') # Deep midnight navy
    fig.patch.set_facecolor('#0B1120')

    # Sophisticated Modern Palette
    c_sky = '#38BDF8'        # Cyan/Sky
    c_teal = '#2DD4BF'       # Teal
    c_indigo = '#818CF8'     # Indigo
    c_purple = '#C084FC'     # Purple
    c_amber = '#FBBF24'      # Amber
    c_emerald = '#34D399'    # Emerald
    c_rose = '#FB7185'       # Rose
    c_card_bg = '#162032'    # Card Slate
    c_card_border = '#2E3F58'# Border
    c_text_bright = '#F8FAFC'# Header White
    c_text_muted = '#94A3B8' # Muted text
    c_text_sub = '#CBD5E1'   # Secondary text

    def draw_container(x, y, w, h, title, subtitle, accent_color):
        # Outer soft container glow & background
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
                              facecolor='#101A2B', edgecolor=accent_color, linewidth=2.0, alpha=0.95, zorder=1)
        ax.add_patch(rect)
        
        # Container header banner
        banner = FancyBboxPatch((x + 0.008, y + h - 0.045), w - 0.016, 0.038,
                                boxstyle="round,pad=0.006,rounding_size=0.012",
                                facecolor=accent_color, edgecolor='none', alpha=0.18, zorder=2)
        ax.add_patch(banner)
        
        ax.text(x + 0.02, y + h - 0.026, title.upper(), color=accent_color, fontsize=12, fontweight='bold',
                va='center', zorder=3, fontfamily='sans-serif')
        ax.text(x + w - 0.02, y + h - 0.026, subtitle, color=c_text_muted, fontsize=9.5, fontstyle='italic',
                ha='right', va='center', zorder=3, fontfamily='sans-serif')

    def draw_card(x, y, w, h, title, items, badge_color=c_sky):
        # Clean rounded component card
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                              facecolor=c_card_bg, edgecolor=c_card_border, linewidth=1.2, zorder=2)
        ax.add_patch(card)
        
        # Accent indicator bar
        line = patches.Rectangle((x + 0.015, y + h - 0.032), 0.005, 0.018, facecolor=badge_color, edgecolor='none', zorder=3)
        ax.add_patch(line)
        
        ax.text(x + 0.026, y + h - 0.023, title, color=c_text_bright, fontsize=10.5, fontweight='bold',
                va='center', zorder=3, fontfamily='sans-serif')
        
        start_y = y + h - 0.052
        for i, item in enumerate(items):
            item_y = start_y - i * 0.024
            bullet = patches.Circle((x + 0.024, item_y), 0.0022, facecolor=badge_color, edgecolor='none', zorder=3)
            ax.add_patch(bullet)
            ax.text(x + 0.034, item_y, item, color=c_text_sub, fontsize=8.8, va='center', zorder=3, fontfamily='sans-serif')

    def draw_arrow(x1, y1, x2, y2, color=c_sky, lw=1.8, rad=0.0):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                connectionstyle=f"arc3,rad={rad}",
                                arrowstyle="->,head_length=5,head_width=3.5",
                                color=color, linewidth=lw, zorder=5)
        ax.add_patch(arrow)

    # -------------------------------------------------------------
    # Main Title
    # -------------------------------------------------------------
    ax.text(0.5, 0.975, "MOL_SMELL : END-TO-END METHODOLOGY FLOWCHART",
            color=c_text_bright, fontsize=18, fontweight='heavy', ha='center', va='center', fontfamily='sans-serif')
    ax.text(0.5, 0.952, "Computational Olfaction • Dual Graph/Fingerprint Representations • SmellGNN Multi-Label Predictor • Inverse SMILES VAE",
            color=c_sky, fontsize=10.5, ha='center', va='center', fontfamily='sans-serif')

    # -------------------------------------------------------------
    # STAGE 1: Data Ingestion & Curation
    # -------------------------------------------------------------
    draw_container(0.04, 0.74, 0.92, 0.19, "Stage 1: Multi-Source Olfaction Data Ingestion & Standardization",
                   "Pyrfume Repositories • Label Harmonization", c_sky)
    
    draw_card(0.06, 0.755, 0.26, 0.125, "Raw Olfaction Corpora", [
        "Pyrfume Leffingwell (3,522 molecules)",
        "GoodScents & Arctander archives",
        "Raw Isomeric SMILES + Odor terms",
        "Sparse sensory behavior tags"
    ], badge_color=c_sky)

    draw_card(0.36, 0.755, 0.28, 0.125, "Standardization & Curation", [
        "RDKit Canonicalization & Sanitization",
        "Deduplication & valence verification",
        "Text normalization: regex cleaning",
        "Filter rare labels (min count ≥ 20) → 113 classes"
    ], badge_color=c_teal)

    draw_card(0.68, 0.755, 0.26, 0.125, "Dataset Partitioning", [
        "MultiLabelBinarizer (113 multi-hot sparse)",
        "Bemis-Murcko Scaffold Split (strict OOD)",
        "Random Train / Val Split (80 / 20)",
        "Class imbalance pos_weight computation"
    ], badge_color=c_amber)

    draw_arrow(0.32, 0.817, 0.36, 0.817, color=c_sky, lw=2.0)
    draw_arrow(0.64, 0.817, 0.68, 0.817, color=c_teal, lw=2.0)
    draw_arrow(0.50, 0.74, 0.50, 0.705, color=c_sky, lw=2.2)

    # -------------------------------------------------------------
    # STAGE 2: Dual Molecular Featurization
    # -------------------------------------------------------------
    draw_container(0.04, 0.505, 0.92, 0.20, "Stage 2: Dual Representation & Structural Featurization",
                   "Bit-Vector Baseline vs. Rich Stereochemical Graph", c_purple)

    draw_card(0.06, 0.52, 0.42, 0.135, "Branch A: Fingerprint Baseline Representation", [
        "Morgan / Extended-Connectivity Fingerprint (ECFP4)",
        "Fixed radius = 2, Bit length = 1024-bit hash vector",
        "Folded bit indices mapped to float32 tensor",
        "Captures circular chemical environments without 3D stereo"
    ], badge_color=c_amber)

    draw_card(0.52, 0.52, 0.42, 0.135, "Branch B: Stereochemical Molecular Graph (PyG Data)", [
        "Atom Nodes X (dim=26): Element, Hybridization, Degree, H-count",
        "Chirality Encoding: Tetrahedral CW / CCW (R/S enantiomers)",
        "Bond Edges E (dim=12): Order (single/double/triple/aromatic)",
        "Stereochemistry: E/Z geometric isomerism, Conjugation, Rings"
    ], badge_color=c_purple)

    draw_arrow(0.27, 0.505, 0.27, 0.47, color=c_amber, lw=2.0)
    draw_arrow(0.73, 0.505, 0.73, 0.47, color=c_purple, lw=2.0)

    # -------------------------------------------------------------
    # STAGE 3: Model Training & Evaluation (Forward Prediction)
    # -------------------------------------------------------------
    draw_container(0.04, 0.27, 0.92, 0.20, "Stage 3: Forward Modeling & Multi-Label Odor Prediction",
                   "Baseline MLP vs. Advanced Attentive SmellGNN", c_emerald)

    draw_card(0.06, 0.285, 0.26, 0.135, "Baseline Model (MLP)", [
        "Input: 1024-bit Morgan Vector",
        "FC Layers (1024 → 512 → 256)",
        "BatchNorm1d, ReLU, Dropout(0.3)",
        "Output: 113 Odor Class Logits",
        "Val Mean AUROC: ~0.874"
    ], badge_color=c_amber)

    draw_card(0.36, 0.285, 0.32, 0.135, "Upgraded SmellGNN (GINEConv + Attention)", [
        "3x GINEConv Message Passing (fuses atom + bond feats)",
        "AttentionalAggregation: focus on odorophores (-SH, esters)",
        "Global Mean Pooling: overall scaffold geometry",
        "Concatenated Readout (2 * hidden_dim) → MLP Head",
        "Val Mean AUROC: ~0.884 | Loss: BCEWithLogits + pos_weight"
    ], badge_color=c_emerald)

    draw_card(0.72, 0.285, 0.22, 0.135, "Evaluation Metrics", [
        "Per-class AUROC (113 labels)",
        "Per-class AUPRC (Average Precision)",
        "Multi-hot ground truth evaluation",
        "Enantiomer differentiation verification"
    ], badge_color=c_teal)

    draw_arrow(0.68, 0.352, 0.72, 0.352, color=c_emerald, lw=2.0)
    # Forward GNN connects to Latent scoring in Stage 4
    draw_arrow(0.52, 0.27, 0.52, 0.235, color=c_emerald, lw=2.2)

    # -------------------------------------------------------------
    # STAGE 4: Inverse Design (SMILES VAE & Latent Optimization)
    # -------------------------------------------------------------
    draw_container(0.04, 0.035, 0.92, 0.20, "Stage 4: Inverse Olfactory Design & Web Application",
                   "Smell-to-Molecule Synthesis via Continuous Latent Exploration & FastAPI Deployment", c_rose)

    draw_card(0.06, 0.05, 0.26, 0.135, "SMILES Seq2Seq VAE", [
        "BiGRU Encoder → Latent Space z (128-dim)",
        "Autoregressive GRU Decoder",
        "Teacher Forcing + Beta-KL Loss",
        "Learns continuous molecular distribution"
    ], badge_color=c_rose)

    draw_card(0.36, 0.05, 0.32, 0.135, "Latent Odor Search & Scoring", [
        "User target scent profile (e.g. Woody=1.0, Fruity=0.8)",
        "Sample latent prior z ~ N(0, I) → Decode candidates",
        "RDKit Chemical Validity & Sanitization Filter",
        "Re-score & rank candidates with trained SmellGNN"
    ], badge_color=c_sky)

    draw_card(0.72, 0.05, 0.22, 0.135, "FastAPI & Interactive UI", [
        "NCI Cactus structure resolver",
        "/api/predict: odor radar spectrum",
        "/api/generate: candidate molecules",
        "Dynamic 3D molecular canvas"
    ], badge_color=c_teal)

    draw_arrow(0.32, 0.117, 0.36, 0.117, color=c_rose, lw=2.0)
    draw_arrow(0.68, 0.117, 0.72, 0.117, color=c_sky, lw=2.0)

    # Final Canvas limits & save
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Flowchart successfully updated and saved to {output_path}")

if __name__ == "__main__":
    create_flowchart("methodology_flowchart.png")
