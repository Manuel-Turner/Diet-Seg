# Changelog

## Reproducibility release

- Added reproducibility environment files.
- Added centralized YAML configs for BraTS2019 and BraTS2020.
- Added fold-wise teacher entropy hardness generation.
- Standardized patch size to 128^3.
- Standardized optimizer to AdamW lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5.
- Added sliding-window inference settings.
- Clarified teacher-student leakage prevention.
- Marked legacy Dice-error hardness code.
