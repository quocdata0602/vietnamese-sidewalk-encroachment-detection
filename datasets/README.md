# Dataset

## Data Sources

| Source | Encroachment | Non-encroachment | Total |
|---|---:|---:|---:|
| Self-collected Vietnamese street images | 2,112 | 547 | 2,659 |
| FootpathVision | 0 | 452 | 452 |
| Roboflow Street Vendors | 496 | 0 | 496 |
| Internet-sourced images | 158 | 0 | 158 |
| **Complete collection** | **2,766** | **999** | **3,765** |

## Data Used for Each Task

| Task | Data sources included | Number of images |
|---|---|---:|
| Binary classification | 1,735 self-collected encroachment images, 547 self-collected non-encroachment images, and 452 FootpathVision non-encroachment images | 2,734 |
| Road–sidewalk segmentation | 1,735 self-collected encroachment images, 547 self-collected non-encroachment images, and 452 FootpathVision non-encroachment images | 2,734 |
| Object detection | 2,112 self-collected encroachment images, 496 Roboflow Street Vendors images, and 158 Internet-sourced images | 2,766 |

## Dataset Splits

| Task | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Binary classification | 1,913 | 410 | 411 | 2,734 |
| Road–sidewalk segmentation | 1,913 | 410 | 411 | 2,734 |
| Object detection | 2,240 | 263 | 263 | 2,766 |

> Models belonging to the same task were trained and evaluated using the same dataset sources and data split.