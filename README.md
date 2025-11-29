# Course on Computer Vision (CAS 2025) 

This repository contains content for a two-day course on the landscape of today’s computer vision — from CNNs to diffusion and beyond. This course was delivered as part of the Certificate of Advanced Studies (CAS) program at the Lucerne University of Applied Sciences (Hochschule Luzern HSLU) in November 2025. 

This course allows participants to get a hands-on, practical introduction to how machines "see" the world. We begin from early concepts of vision to the multimodal vision-language models. On this journey, we cover discriminative models (CNN-era) to generative models (diffusion, vision transformer).


## Table of Contents

- [What You’ll Learn](#what-youll-learn)
- [List of Slides & Notebooks](#list-of-slides--notebooks)
- [Setup](#-setup)
- [License](#license)
- [Citation](#citation)
- [Contributing](#contributing)

## 🔍 What You’ll Learn

- Evolution of computer vision and core problem types
- CNNs for image classification
- Object detection 
- Semantic segmentation
- Vision Generative models landscape
- Autoregressive model
- GANs, VAEs

---

## List of Slides & Notebooks

**Day 1 – From CNN basics to segmentation**

| Topic | Slides | Notebook | Colab |
| --- | --- | --- | --- |
| Test Your Setup | [link](https://docs.google.com/presentation/d/1uXYZ74svgoD35b9szrZQe1RSNj1Oe1_R0tEipTTB1Cw/edit?usp=share_link) | [notebooks/day1/00_setup_test.ipynb](notebooks/day1/00_setup_test.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exomishra/computer-vision-cas-hslu-2025/blob/main/notebooks/day1/00_setup_test.ipynb) |
| Image Data Basics | [link](https://docs.google.com/presentation/d/1q94C5oCHwC5wjIv1g2qcdDZ9Czm6nfqu0_dXzVhFVvQ/edit?usp=share_link) | [notebooks/day1/01_image_data_basics.ipynb](notebooks/day1/01_image_data_basics.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exomishra/computer-vision-cas-hslu-2025/blob/main/notebooks/day1/01_image_data_basics.ipynb) |
| CNN Intro & Training | [link](https://docs.google.com/presentation/d/1CwfCCb-Q71MQ57v1VU2U35hbfon1Oufd-M7oAnn0jCw/edit?usp=share_link) | [notebooks/day1/02_cnn_intro_and_training.ipynb](notebooks/day1/02_cnn_intro_and_training.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exomishra/computer-vision-cas-hslu-2025/blob/main/notebooks/day1/02_cnn_intro_and_training.ipynb) |
| Object Detection (YOLOv8) | [link](https://docs.google.com/presentation/d/1ufM6Uekk-HYiQcZ2VyTMlOZXm_EnYa6Gw4vZ45FujdM/edit?usp=share_link) | [notebooks/day1/03_object_detection.ipynb](notebooks/day1/03_object_detection.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exomishra/computer-vision-cas-hslu-2025/blob/main/notebooks/day1/03_object_detection.ipynb) |
| Semantic Segmentation | [link](https://docs.google.com/presentation/d/1It3G76HavOVgkxd6axaW618UblgahuG_MeW8J8XUOOo/edit?usp=share_link) | [notebooks/day1/04_semantic_segmentation.ipynb](notebooks/day1/04_semantic_segmentation.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exomishra/computer-vision-cas-hslu-2025/blob/main/notebooks/day1/04_semantic_segmentation.ipynb) |

**Day 2 – Generative Models**

| Topic | Slides | Notebook | Colab |
| --- | --- | --- | --- |
| PixelCNN & Autoregressive Models | [link](https://docs.google.com/presentation/d/1jouvCv8npiLuXdCqhtwizKnAyEeU788z3b1dRxfQIu4/edit?usp=sharing) | [notebooks/day2/01_pixelcnn.ipynb](notebooks/day2/01_pixelcnn.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exomishra/computer-vision-cas-hslu-2025/blob/main/notebooks/day2/01_pixelcnn.ipynb) |
| Variational Autoencoders | [link](https://docs.google.com/presentation/d/1Rtx3-bDWTpo8PhPxKbeLfd0mh_GRsMZSwD7-pXcnl_I/edit?usp=sharing) | [notebooks/day2/02_vae.ipynb](notebooks/day2/02_vae.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exomishra/computer-vision-cas-hslu-2025/blob/main/notebooks/day2/02_vae.ipynb) |
| Generative Adversarial Networks | [link](https://docs.google.com/presentation/d/1mfI9AXM4m9He40_mAJ-Z1bZU0dJr3HScsj4rQNahrZc/edit?usp=sharing) | [notebooks/day2/03_gans.ipynb](notebooks/day2/03_gans.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exomishra/computer-vision-cas-hslu-2025/blob/main/notebooks/day2/03_gans.ipynb) |

---

## 💻 Setup

### Local Setup (with `uv`)

- Clone this repository:
```sh
git clone https://github.com/exomishra/computer-vision-cas-hslu-2025.git
```

- Install `uv` (if you don't already have it):

    `uv` is a super-fast Python package manager. It will install all necessary libraries and also ensure what we do here does not affect your machine by creating a virtual environment. Steps to install uv depend on your machine (mac/linux/windows). Please see [here](https://docs.astral.sh/uv/getting-started/installation). Use any one of the following (in your [terminal](https://www.freecodecamp.org/news/command-line-for-beginners/)):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh

# OR

wget -qO- https://astral.sh/uv/install.sh | sh

# OR 

pip install uv

# OR ON MAC
brew install uv
```

- Create and sync your virtual environment:

```sh
uv venv
uv sync
```

- Activate your virtual environment:

```sh
# linux / macos
source .venv/bin/activate

# windows
.venv\Scripts\activate
```

- Running Notebooks:

    If you have an IDE and know how to run a notebook, do that. If you don't know what that means, run this:

```sh
uv run --with jupyter jupyter lab
```

- Test your setup by running the [00_setup_test](notebooks/day1/00_setup_test.ipynb) notebook.

---

## License
- Code: [MIT License](./LICENSE)  
- Teaching material (slides, docs, etc.): [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)

--- 


## Citation

If you use these materials in your research or teaching, please cite as:

**Text Citation:**
> Mishra, L. (2025). *Course on Computer Vision (CAS 2025)*. Lucerne University of Applied Sciences and Arts (HSLU). https://github.com/exomishra/computer-vision-cas-hslu-2025

**BibTeX:**
```bibtex
@misc{mishra2025cv,
  author       = {Lokesh Mishra},
  title        = {Course on Computer Vision (CAS 2025)},
  year         = {2025},
  howpublished = {\url{https://github.com/exomishra/computer-vision-cas-hslu-2025}},
  note         = {Lucerne University of Applied Sciences and Arts (HSLU)}
}
```

---

### Repo Structure

```
.
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── configs/
│   └── day1_cnn.yaml
├── data/
│   ├── face_samples/
│   └── MNIST/
│       └── raw/ (idx file dumps)
├── notebooks/
│   ├── day1/
│   │   ├── 00_setup_test.ipynb
│   │   ├── 01_image_data_basics.ipynb
│   │   ├── 02_cnn_intro_and_training.ipynb
│   │   ├── 03_object_detection.ipynb
│   │   └── 04_semantic_segmentation.ipynb
│   └── day2/
│       ├── 01_pixelcnn.ipynb
│       ├── 02_vae.ipynb
│       ├── 03_gans.ipynb
│       └── 04_research_trends.ipynb
├── slides/
│   └── slides.md
└── src/
    └── cvcourse/
  ├── __init__.py
  ├── config.py
  ├── data/
  ├── models/
  ├── training/
  ├── utils/
  └── viz/

```

---

## Contributing

Contributions are welcome! If you have suggestions, corrections, or new material to add, please open an issue or submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to follow the code of conduct and respect the collaborative spirit of this project.

---

### Package Layout & Teaching Philosophy

- **`src/cvcourse`** holds all reusable logic (data, models, training, visualization) with docstrings so students can read code like prose.
- **`configs/`** contains YAML presets that notebooks load via `cvcourse.config.load_config`, ensuring every run is reproducible.
- **`notebooks/`** stay as orchestration/storytelling layers — they import from `cvcourse` instead of redefining helpers inline.
- **`uv`** is the canonical package manager. Use `uv venv && uv sync` to reproduce the environment and `uv run --with jupyter jupyter lab` when you need a notebook server.