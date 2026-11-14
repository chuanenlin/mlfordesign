# Machine Learning for Design: notebooks

Colab notebooks for the workshops in [Machine Learning for Design](https://mlfordesign.com). You can open them from the course schedule, then save a copy to your Drive (File → Save a copy in Drive) before you run them. Each notebook starts with an example about the 56 museum chairs, and the example's images and figures are saved in it, so you can see the results first and then try your own work.

Each notebook says what it needs in its first cell. The weeks 3, 5 and 7 notebooks use the course Replicate token, and weeks 7 and 8 use the course Gemini key, which the instructor shares in the first studio. The week 5 slider and week 6 tool notebooks also need an A100 runtime and your own Hugging Face token.

| Week | Notebook |
|---|---|
| 2 | [Build a mood board](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-02-moodboard.ipynb) |
| 3 | [Sketch to image](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-03-sketch.ipynb) |
| 4 | [Map your set](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-04-map-your-set.ipynb) |
| 5 | [Train and compare two models](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-05-train.ipynb) |
| 5 | [Make a slider for your model](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-05-sliders.ipynb) |
| 6 | [A tool around your model](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-06-tool.ipynb) |
| 7 | [Read an agent's log](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-07-agent-log.ipynb) |
| 8 | [Ask simulated participants](https://colab.research.google.com/github/chuanenlin/mlfordesign/blob/main/notebooks/week-08-simulated-users.ipynb) |

## Example: museum chairs

`examples/museum-chairs/train` has the 56 chair photos from week 5, from the open-access collections of The Metropolitan Museum of Art and the Art Institute of Chicago (CC0). `credits.csv` lists each chair with its museum record. `examples/museum-chairs/made` has four images from the FLUX.1 [dev] LoRA trained on them, with the trigger word MUSCHR. `datasheet.md` answers the questions from Gebru et al.'s "Datasheets for Datasets" for this set, as in week 4. `captions-museum.csv` has a caption for each chair written from its museum record, which the week 4 notebook compares with automatic captions.

`examples/mood-pool` has 294 public-domain images from the Art Institute of Chicago for the week 2 mood boards (rooms, textiles, ceramics, prints and more), with `credits.csv` listing each one's record. Together with the 56 chairs, they make the pool of 350 images in week 2.

`examples/chair-sketch` has the rough chair sketch from week 3, and the same sketch with the legs and back posts redrawn as outlined bars.

`examples/chair-tool` has the week 6 slider tool, `app.py`, and `CLAUDE.md`, the instruction file from the week 6 slides. The tool has no Make strip button yet, so you can ask your coding agent to add one, as in the slides. `examples/museum-chairs/trainings.json` lists the two trainings behind the week 5 example, so the train notebook shows their results without training again.
