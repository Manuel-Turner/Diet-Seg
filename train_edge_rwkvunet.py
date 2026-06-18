"""Legacy entry point kept for backward compatibility.

The reproducibility release uses scripts/train_student.py with centralized YAML
configuration. This wrapper preserves the old filename while removing the
inconsistent img_size=160, img_channels=160, batch_size=8, max_epochs=400
defaults.
"""

from scripts.train_student import main


if __name__ == "__main__":
    main()
