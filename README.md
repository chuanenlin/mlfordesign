# Machine Learning for Design: notebooks

Colab notebooks for the workshops in [Machine Learning for Design](https://mlfordesign.com). You can open them from the course schedule, then save a copy to your Drive (File, then Save a copy in Drive) before you run them. Each one starts with the museum chairs example, and the saved outputs show its results on those chairs before you try your own images.

| Week | Notebook |
|---|---|
| 4 | [Map your set](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-04-map-your-set.ipynb) |
| 5 | [Map your images](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-05-map.ipynb) |
| 5 | [Make a slider for your model](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-05-sliders.ipynb) |

## Example: museum chairs

`examples/museum-chairs/train` has the 56 chair photos from week 5, from the open-access collections of The Metropolitan Museum of Art and the Art Institute of Chicago (CC0). `credits.csv` lists each chair with its museum record. `examples/museum-chairs/made` has four images from the FLUX.1 [dev] LoRA trained on them, with the trigger word MUSCHR.
