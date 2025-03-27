# NoMUR

NoMUR stands for Northern Mindanao Unstructured Roads. It is a self-gathered dataset containing NoMURLane, NoMURPanel, and NoMURDistress.

- NoMURLane: **No**rthern **M**indanao **U**nstructured **R**oads – a four-point polygon-annotated, single-class dataset for Road **Lane** segmentation.
- NoMURPanel: **No**rthern **M**indanao **U**nstructured **R**oads – a four-point polygon-annotated, single-class dataset for Pavement **Panel** segmentation.
- NoMURDistress: **No**rthern **M**indanao **U**nstructured **R**oads – a multi-class dataset for Pavement Panel **Distress** classification.

This repository contains utility functions for processing the dataset. 

## Data Augmentation

You can perform data augmentation for segmentation task on NoMURLane and NoMURPanel datasets, where each annotation is a 4-point polygon.

```
python augment.py --data config.yaml
```

Modify the `config.yaml` file to fill in appropriate parameters, such as source path and destination path. 