import gradio as gr
from ultralytics import YOLO
import os

# catgories
format =  { 0: 'Adenocarcinoma case',
            1: 'Bengin case',
            2: 'Large cell Carcinoma case',
            3: 'Malignant case',
            4: 'Normal case',
            5: 'Squamous cell Carcinoma case'}

# returning classifiers output
def image_classifier(inp):
    model = YOLO("best-2.pt")

    result = model.predict(source=inp)
    probs = result[0].probs
    max_tensor = max(probs)
    tensor_pos = ((probs == max_tensor).nonzero(as_tuple=True)[0])

    return format.get(int(tensor_pos))

# gradio code block for input and output
with gr.Blocks() as app:
    gr.Markdown("## Lung Cancer classification using Yolov8")
    with gr.Row():
        inp_img = gr.Image()
        out_txt = gr.Textbox()
    btn = gr.Button(value="Submit")
    btn.click(image_classifier, inputs=inp_img, outputs=out_txt)

    gr.Markdown("## Image Examples")
    gr.Examples(
        examples=[os.path.join(os.path.dirname(__file__), "1.jpg"), os.path.join(os.path.dirname(__file__), "2.jpg")],
        inputs=inp_img,
        outputs=out_txt,
        fn=image_classifier,
        cache_examples=True,
    )

app.launch(share=True)
