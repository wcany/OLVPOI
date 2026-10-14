import argparse
from pathlib import Path


NEW_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = NEW_ROOT / "data" / "original_data_p6(1).xlsx"


def parameter_parser():
    parser = argparse.ArgumentParser()

    # Training parameters.
    parser.add_argument("--epoch", dest="epochs", type=int, default=200)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=0.001 )   # 0.001
    parser.add_argument("--weight-decay", type=float, default=5e-3) # 5e-3
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--alpha", type=float, default=0.8, help="Reconstruction loss weight.")
    parser.add_argument("--beta", type=float, default=0.2,  help="Contrastive loss weight.")
    parser.add_argument("--threshold", dest="classification_threshold", type=float, default=0.5)

    # Network parameters.
    parser.add_argument("--hidden-dim", type=int, default=64) # 64
    parser.add_argument("--latent_dim", type=int, default=16) # 16
    parser.add_argument("--dropout", type=float, default=0.1)

    # Fixed Step1 KNN construction.
    parser.add_argument("--k", type=int, default=5)  # 5

    # Fixed loss and scheduler settings.
    parser.add_argument("--contrastive-temperature", type=float, default=0.5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=50)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)

    # Independent random seeds.
    parser.add_argument("--cv-seed", type=int, default=42)  # 42
    parser.add_argument("--model-seed", type=int, default=0) # 0

    parser.add_argument("--device", dest="cuda_device", type=str, default="0")
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--no-progress",dest="show_progress",action="store_false",default=True)
    parser.add_argument("--path", dest="data_path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-dir", type=Path, default=NEW_ROOT / "results")
    
    return parser.parse_args()
