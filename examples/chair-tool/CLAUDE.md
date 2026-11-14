# Chair slider tool

app.py is a small Gradio tool around one fine-tuned image model (FLUX.1 [dev] with a LoRA). make_image() makes one chair with the slider at strength s.

- Keep make_image() and encode() as they are. They are the same slider code as the week 5 notebook, so editing them would make every chair different from the earlier results.
- Keep the tool on one page, with the controls on the left and the results on the right.
- Don't load the model or run the tool to test a change. You can check it with python3 -m py_compile app.py.
- Give each new control a plain label that says what it does.
