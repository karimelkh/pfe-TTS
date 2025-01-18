import os

import gradio as gr
import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"

MODELNAME = "tts_models/en/ljspeech/fast_pitch"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
AUDIO_FILENAME = os.path.join(OUTPUTS_DIR, "sample.wav")


def generate_audio(text="Hello, World!"):
    tts = TTS(model_name=MODELNAME).to(device)
    tts.tts_to_file(text=text, file_path=AUDIO_FILENAME)
    return AUDIO_FILENAME

if __name__ == "__main__":
    if not(os.path.exists(OUTPUTS_DIR)):
        os.mkdir(OUTPUTS_DIR)
    demo = gr.Interface(
        fn=generate_audio,
        inputs=[gr.Text(label="Text"),],
        outputs=[gr.Audio(label="Audio"),],
    )

    demo.launch()
