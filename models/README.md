# Models

This folder contains the trained models used by the sidewalk encroachment analysis system.

| File | Purpose | Classes |
|---|---|---|
| `best_object_detection.pt` | Detect sidewalk encroachment objects | `ad_board`, `bin`, `car`, `motorbike`, `table_chair`, `umbrella`, `vendor_cart` |
| `best_road_sidewalk_seg.pt` | Segment road and sidewalk regions | `road`, `sidewalk` |

The model file names should not be changed unless the corresponding paths in `app.py` and `experiments/experiments.ipynb` are also updated.