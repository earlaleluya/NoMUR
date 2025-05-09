# NoMUR

NoMUR stands for Northern Mindanao Unstructured Roads. It is a self-gathered dataset containing NoMURLane, NoMURPanel, and NoMURDistress.

- NoMURLane: **No**rthern **M**indanao **U**nstructured **R**oads – a four-point polygon-annotated, single-class dataset for Road **Lane** segmentation.
- NoMURPanel: **No**rthern **M**indanao **U**nstructured **R**oads – a four-point polygon-annotated, single-class dataset for Pavement **Panel** segmentation.
- NoMURDistress: **No**rthern **M**indanao **U**nstructured **R**oads – a multi-class dataset for Pavement Panel **Distress** classification.


# Citations

NoMURLane:
```
@data{NoMURLane,
    author = {Aleluya, Earl Ryan and Maluya, Melody Mae and Miranda, Genesis and Barro, Shiela Mae and Clar, Steve and Alagon, Francis Jann and Aleluya, Edward Rowe and Opon, Joel and Salaan, Carl John},
    publisher = {Harvard Dataverse},
    title = {{NoMURLane}},
    year = {2025},
    version = {V1},
    doi = {10.7910/DVN/BAMYR1},
    url = {https://doi.org/10.7910/DVN/BAMYR1}
}
```

NoMURPanel:
```
@data{NoMURPanel,
    author = {Aleluya, Earl Ryan and Maluya, Melody Mae and Miranda, Genesis and Barro, Shiela Mae and Clar, Steve and Alagon, Francis Jann and Aleluya, Edward Rowe and Opon, Joel and Salaan, Carl John},
    publisher = {Harvard Dataverse},
    title = {{NoMURLane}},
    year = {2025},
    version = {V1},
    doi = {10.7910/DVN/BAMYR1},
    url = {https://doi.org/10.7910/DVN/BAMYR1}
    }
```

NoMURDistress:
```
@data{NoMURDistress,
    author = {Aleluya, Earl Ryan and Maluya, Melody Mae and Miranda, Genesis and Barro, Shiela Mae and Clar, Steve and Alagon, Francis Jann and Aleluya, Edward Rowe and Opon, Joel and Salaan, Carl John},
    publisher = {Harvard Dataverse},
    title = {{NoMURDistress}},
    year = {2025},
    version = {V1},
    doi = {10.7910/DVN/HUEFOS},
    url = {https://doi.org/10.7910/DVN/HUEFOS}
    }
```


## Dependencies

Create a virtual environment and install dependencies by:
```
pip install -r requirements.txt
```



## Data Augmentation
This repository contains utility functions for processing the dataset. You can perform data augmentation for segmentation task on NoMURLane and NoMURPanel datasets, where each annotation is a 4-point polygon.

```
python augment.py --data config.yaml 
```

If you do not want to visualize augmentation per image, you can add argument `no-show`.

```
python augment.py --data config.yaml --no-show
```

Modify the `config.yaml` file to fill in appropriate parameters, such as `source_path`, `destination_path`, `img_size`, `id2class`, and `outputs_per_img`.