# Normal-NeRF: Ambiguity-Robust Normal Estimation for Highly Reflective Scenes

## Installation

Normal-NeRF is built on [nerfstudio](https://docs.nerf.studio/). Follow their [installation guide](https://docs.nerf.studio/quickstart/installation.html) to create the environment and install dependencies.

Once you have finished installing nerfstudio, you can install Normal-NeRF using the following commands:
```
git clone https://github.com/sjj118/Normal-NeRF.git
cd Normal-NeRF
pip install ./griddecay
pip install -e .
```

## Dataset

Download datasets from following links: [NeRF Synthetic](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4), [Shiny Blender](https://storage.googleapis.com/gresearch/refraw360/ref.zip) and [Glossy Synthetic](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/yuanly_connect_hku_hk/EvNz_o6SuE1MsXeVyB0VoQ0B7L4i2I1noCWGs2qu_S23BA?e=6fzVfH).

To preprocess the Shiny Blender dataset, run:
```
python preprocess_shiny.py data/shiny_blender
```

For Glossy Synthetic dataset, download both `GlossySynthetic.tar.gz` and `glossy-synthetic-nvs.zip`, then extract their contents into the same directory. For example:
```
GlossySynthetic
├── angel
├── angel_nvs
├── bell
├── bell_nvs
├── ...
```
Next, convert the dataset format by running the following command:
```
python preprocess_glossy.py data/GlossySynthetic data/glossy_synthetic
```

## Running

```
# Training
ns-train normalnerf --data data/nerf_synthetic/ship

# Evaluation
ns-eval --load-config <path to experiment's config.yml>

# Rendering
ns-render dataset --image-format png --load-config <path to experiment's config.yml>
```