# Machine Learning for Design: notebooks

Colab notebooks for the workshops in [Machine Learning for Design](https://mlfordesign.com). You can open them from the course schedule, then save a copy to your Drive (File, then Save a copy in Drive) before you run them. Each one starts with an example about chairs, and the saved outputs show its results before you try your own work.

| Week | Notebook |
|---|---|
| 2 | [Group your notes](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-02-cluster-notes.ipynb) |
| 3 | [Sketch to image](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-03-sketch.ipynb) |
| 4 | [Map your set](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-04-map-your-set.ipynb) |
| 5 | [Train and compare two models](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-05-train.ipynb) |
| 5 | [Make a slider for your model](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-05-sliders.ipynb) |
| 6 | [A tool around your model](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-06-tool.ipynb) |
| 7 | [Read an agent's log](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-07-agent-log.ipynb) |
| 8 | [Ask simulated users](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-08-simulated-users.ipynb) |

## Example: museum chairs

`examples/museum-chairs/train` has the 56 chair photos from week 5, from the open-access collections of The Metropolitan Museum of Art and the Art Institute of Chicago (CC0). `credits.csv` lists each chair with its museum record. `examples/museum-chairs/made` has four images from the FLUX.1 [dev] LoRA trained on them, with the trigger word MUSCHR. `datasheet.md` answers the questions from Gebru et al.'s "Datasheets for Datasets" for this set, as in week 4. `captions-museum.csv` has a caption for each chair written from its museum record, which the week 4 notebook compares with automatic captions.

`examples/chair-sketch` has the rough chair sketch from week 3, and the same sketch with the legs and back posts redrawn as outlined bars.
