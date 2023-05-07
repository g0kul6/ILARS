import gradio as gr
import numpy as np
import cv2 
import os,glob
import json

with gr.Blocks() as demo:
    video_upload = gr.UploadButton(label="Upload the Video", file_types=["video"])
    frames = []
    def get_frame(video):
        frames.clear()
        files = glob.glob('frames/*')
        for f in files:
            os.remove(f)
        cap = cv2.VideoCapture(video.name)
        i = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frames.append(frame)
            # cv2.imwrite("frames/frame_{}.jpeg".format(i),frame)
            i += 1
        cap.release()
        cv2.destroyAllWindows()
    def return_frame(index):
        # img = cv2.imread("frames/frame_{}.jpeg".format(index))
        img = frames[index]
        return img
    slider = gr.Slider(maximum=10000,interactive=True,steps=1)
    video_upload.upload(fn=get_frame, inputs=[video_upload])
    slider.change(return_frame,slider,gr.Image())    
    question = gr.Textbox(label="Question")
    model_type = gr.CheckboxGroup(["SurgGPT","LCGN"],label="Model Choice")
    answer = gr.Textbox(label="Answer")
    predict = gr.Button(value="Predict")
    
if __name__ == "__main__":
    demo.launch()
