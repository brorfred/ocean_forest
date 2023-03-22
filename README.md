# ocean_forest

ocean_forest is a Python package to perform Random Forest regressions based on in-situ obervations of ocean properties. The main use case is Primary Production based on insitu data from Mattei.

## Installation

The easiest approach to install dependencies is to use conda. Just create a virtual envirnment from the included environment.yml file:

```bash
conda env create -f environment.yml 
```

## Usage

```python
import export_production

# Load Mouw data
df = export_production.load()
# Fit a Random forest model to data
model = export_production.regress(df=df)
# Hyper parameters and other presets are stored in the 'settings.toml' file 
# Existing default models are saved in the subfolder rf_models

```

