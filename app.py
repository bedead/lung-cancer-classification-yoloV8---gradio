import gradio as gr
from ultralytics import YOLO

format = { 0: 'Bengin case',
             1: 'Bengin case Malignant case',
             2: 'Malignant case',
             3: 'Malignant case Normal case',
             4: 'Normal case'}

def image_classifier(inp):
    model = YOLO("best.pt")

    result = model.predict(source=inp)
    probs = result[0].probs
    max_tensor = max(probs)
    tensor_pos = ((probs == max_tensor).nonzero(as_tuple=True)[0])
    
    return format.get(int(tensor_pos))

web = gr.Interface(fn=image_classifier, inputs="image", outputs="text")



with gr.Blocks() as site:
    gr.Markdown("Lung cancer detection using Yolov8 model")
    with gr.Row():
        img_input = gr.Image()
        txt_output = gr.Textbox()
    submit_btn = gr.Button("Submit")

    submit_btn.click(image_classifier, inputs=img_input, outputs=txt_output)

web.launch(share=True)

