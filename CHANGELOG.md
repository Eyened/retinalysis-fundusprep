# Changelog

## 1.1.0 (2026-07-10)

### Added
- Full-frame image detection (`is_full_frame`, `CFIBounds.full_frame`)
- Grayscale and RGBA support for contrast enhancement
- Optional `dicom` extra with JPEG decode helpers

### Changed
- More robust bounds extraction for difficult and full-frame images
- `get_cfi_bounds()` falls back to full-frame bounds on failure instead of raising
- Updated regression reference for new algorithm outputs

### Fixed
- Contrast enhancement and plotting for non-RGB inputs
- Mask extraction edge cases (low circle fraction, steep lines)
