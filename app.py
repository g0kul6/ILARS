import os
import cv2
import gradio as gr 
from app_eval import eval


with gr.Blocks(css="footer {visibility: hidden}", title="Surgical-AI") as demo:
    # Heading and authors
    gr.Markdown("""
    <h1 style="text-align: center;">Multimodal Reasoning and Language Prompting for Interactive Learning Assistant for Robotic Surgery. </h1>
    <h3 style="text-align: center;">Gokul Kannan*, Lalithkumar Seenivasan*, Hongliang Ren** </h3>
    <h3 style="text-align: center;">labren.org </h3>
    """)
    # Video Upload Button
    video_upload = gr.UploadButton(label="Upload the Video", file_types=["video"])
    # example video
    video = gr.Video(visible=False)
    # slider
    slider = gr.Slider(maximum=200,interactive=True,steps=1)
    # image
    frame_gr = gr.Image(shape=(1280,720))
    # frames list
    frames = []
    # function to get individual frames form video
    def get_frame(video):
        frames.clear()
        if str(type(video)) == "<class 'str'>":
            cap = cv2.VideoCapture(video)
        else:
            cap = cv2.VideoCapture(video.name)
        i = 0
        for i in range(201):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if ret == False:
                break
            frames.append(frame)
            i += 1
        cap.release()
        cv2.destroyAllWindows()
    # Examples 
    gr.Examples(
        examples=["/home/gokul/g0kul6/ILARS/video1.mp4"],
        inputs=video,
        fn=get_frame,
        outputs = video_upload,
        cache_examples=True,
    )
    # if upload button is triggered 
    video_upload.upload(fn=get_frame, inputs=[video_upload])
    # function to return slider based indexed frame
    def return_frame(index):
        img = frames[index]
        return img
    # if slider is moved
    slider.change(return_frame,slider,frame_gr)    
    # question text box
    question = gr.Textbox(label="Question")
    # model option checkbox
    model_type = gr.CheckboxGroup(["Surgical-MmR"],label="Model Choice")
    # answer textbox
    answer = gr.Textbox(label="Answer")
    # LLM description textbox
    description = gr.Textbox(label="Description")
    # Predict Button
    predict = gr.Button(value="Predict")
    # predict ans using VQA model
    def predict_ans(index,question,model_choice):
        img = frames[index]
        ans,description = eval(model_ver="efvlegpt2rs18",img=img,question=question)
        return ans,description
    # predict button is triggered
    predict.click(fn=predict_ans,inputs=[slider,question,model_type],outputs=[answer,description])
    
# launch the gradio app
if __name__ == "__main__":
    demo.launch(share=True)
