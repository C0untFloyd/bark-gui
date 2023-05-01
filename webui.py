from cProfile import label
from distutils.command.check import check
import gradio as gr
import os
import re
import sys
import numpy as np
import logging
import torch

from bark import SAMPLE_RATE, generate_audio, set_seed
from bark.clonevoice import clone_voice
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
from scipy.io.wavfile import write as write_wav
from datetime import datetime
from time import time
from tqdm.auto import tqdm

OUTPUTFOLDER = "Outputs"

# Most of the chunked generation ripped from https://github.com/serp-ai/bark-with-voice-clone
def split_and_recombine_text(text, desired_length=100, max_length=150):
    # from https://github.com/neonbjb/tortoise-tts
    """Split text it into chunks of a desired length trying to keep sentences intact."""
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in "!?.\n " and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in "!?\n" or (c == "." and peek(1) in "\n ")):
            # seek forward if we have consecutive boundary markers but still within the max length
            while (
                pos < len(text) - 1 and len(current) < max_length and peek(1) in "!?."
            ):
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in "\n ":
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r"^[\s\.,;:!?]*$", s)]

    return rv

def generate_with_settings(text_prompt, semantic_temp=0.7, semantic_top_k=50, semantic_top_p=0.95, coarse_temp=0.7, coarse_top_k=50, coarse_top_p=0.95, fine_temp=0.5, voice_name=None, use_semantic_history_prompt=True, use_coarse_history_prompt=True, use_fine_history_prompt=True, output_full=False):

    # generation with more control
    x_semantic = generate_text_semantic(
        text_prompt,
        history_prompt=voice_name if use_semantic_history_prompt else None,
        temp=semantic_temp,
        top_k=semantic_top_k,
        top_p=semantic_top_p
    )

    x_coarse_gen = generate_coarse(
        x_semantic,
        history_prompt=voice_name if use_coarse_history_prompt else None,
        temp=coarse_temp,
        top_k=coarse_top_k,
        top_p=coarse_top_p
    )
    x_fine_gen = generate_fine(
        x_coarse_gen,
        history_prompt=voice_name if use_fine_history_prompt else None,
        temp=fine_temp
    )

    if output_full:
        full_generation = {
            'semantic_prompt': x_semantic,
            'coarse_prompt': x_coarse_gen,
            'fine_prompt': x_fine_gen
        }
        return full_generation, codec_decode(x_fine_gen)
    return codec_decode(x_fine_gen)

def generate_text_to_speech(text, selected_speaker, text_temp, waveform_temp, quick_generation, complete_settings, progress=gr.Progress(track_tqdm=True)):
    if text == None or len(text) < 1:
        raise gr.Error('No text entered!')

    # Chunk the text into smaller pieces then combine the generated audio

    # generation settings
    if selected_speaker == 'None':
        selected_speaker = None

    voice_name = selected_speaker

    semantic_temp = text_temp
    semantic_top_k = 50
    semantic_top_p = 0.95

    coarse_temp = waveform_temp
    coarse_top_k = 50
    coarse_top_p = 0.95

    fine_temp = 0.5

    use_semantic_history_prompt = "Use semantic history" in complete_settings
    use_coarse_history_prompt = "Use coarse history" in complete_settings
    use_fine_history_prompt = "Use fine history" in complete_settings
    use_last_generation_as_history = "Use last generation as history" in complete_settings
    progress(0, desc="Generating")
    texts = split_and_recombine_text(text)

    all_parts = []
    for i, text in tqdm(enumerate(texts), total=len(texts)):
        set_seed(432)
        if quick_generation == True:
            print(f"\nGenerating Text ({i+1}/{len(texts)}) -> `{text}`")
            audio_array = generate_audio(text, selected_speaker, text_temp, waveform_temp)
            i+=1

        else:
            print(f"\nGenerating Text ({i+1}/{len(texts)}) -> `{text}`")
            full_generation, audio_array = generate_with_settings(
                text,
                semantic_temp=semantic_temp,
                semantic_top_k=semantic_top_k,
                semantic_top_p=semantic_top_p,
                coarse_temp=coarse_temp,
                coarse_top_k=coarse_top_k,
                coarse_top_p=coarse_top_p,
                fine_temp=fine_temp,
                voice_name=voice_name,
                use_semantic_history_prompt=use_semantic_history_prompt,
                use_coarse_history_prompt=use_coarse_history_prompt,
                use_fine_history_prompt=use_fine_history_prompt,
                output_full=True,
            )
            i+=1
        set_seed(-1)
        if len(texts) > 1:
            save_wav(audio_array, create_filename(OUTPUTFOLDER, "audioclip",".wav"))

        if quick_generation == False & use_last_generation_as_history:
            # save to npz
            voice_name = create_filename(OUTPUTFOLDER, "audioclip", "")
            save_voice(voice_name,
                      full_generation['semantic_prompt'],
                      full_generation['coarse_prompt'],
                      full_generation['fine_prompt'])
            # loading voice from custom folder needs to have extension
            voice_name = voice_name + ".npz"
        all_parts.append(audio_array)

    audio_array = np.concatenate(all_parts, axis=-1)

    # save & play audio
    result = create_filename(OUTPUTFOLDER, "final",".wav")
    save_wav(audio_array, result)
    return result

def create_filename(path, name, extension):
    now = datetime.now()
    date_str = now.strftime("%m-%d-%Y")
    outputs_folder = os.path.join(os.getcwd(), path)
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    sub_folder = os.path.join(outputs_folder, date_str)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    now = datetime.now()
    time_str = now.strftime("%H-%M-%S")
    file_name = f"{name}_{time_str}{extension}"
    return os.path.join(sub_folder, file_name)


def save_wav(audio_array, filename):
    write_wav(filename, SAMPLE_RATE, audio_array)

def save_voice(filename, semantic_prompt, coarse_prompt, fine_prompt):
    np.savez_compressed(
        filename,
        semantic_prompt=semantic_prompt,
        coarse_prompt=coarse_prompt,
        fine_prompt=fine_prompt
    )
    

def on_quick_gen_changed(checkbox):
    if checkbox == False:
        return gr.CheckboxGroup.update(visible=True)
    return gr.CheckboxGroup.update(visible=False)



logger = logging.getLogger(__name__)

autolaunch = False

if len(sys.argv) > 1:
    autolaunch = "-autolaunch" in sys.argv


if torch.cuda.is_available() == False:
    os.environ['BARK_FORCE_CPU'] = 'True'
    logger.warning("No CUDA detected, fallback to CPU!")

print(f'smallmodels={os.environ.get("SUNO_USE_SMALL_MODELS", False)}')
print(f'enablemps={os.environ.get("SUNO_ENABLE_MPS", False)}')
print(f'offloadcpu={os.environ.get("SUNO_OFFLOAD_CPU", False)}')
print(f'forcecpu={os.environ.get("BARK_FORCE_CPU", False)}')
print(f'autolaunch={autolaunch}\n')



# Collect all existing speakers/voices in dir
speakers_list = ['None']
for file in os.listdir("./bark/assets/prompts"): 
    if file.endswith(".npz"):
        speakers_list.append(file[:-4])

# Create Gradio Blocks

with gr.Blocks(title="Bark Enhanced Gradio GUI", mode="Bark Enhanced") as barkgui:
    gr.Markdown("### [Bark Enhanced](https://github.com/C0untFloyd/bark-gui)")
    with gr.Tab("TTS"):
        placeholder = """Enter text here. Special meanings: [laughter] [laughs] [sighs] [music] [gasps] [clears throat]
— or ... for hesitations
♪ for song lyrics"""
        input_text = gr.Textbox(label="Input Text", lines=4, placeholder=placeholder)
        speaker = gr.Dropdown(speakers_list, value=speakers_list[0], label="Voice")
        text_temp = gr.Slider(
            0.1,
            1.0,
            value=0.7,
            label="Generation Temperature",
            info="1.0 more diverse, 0.1 more conservative",
        )
        waveform_temp = gr.Slider(0.1, 1.0, value=0.7, label="Waveform temperature", info="1.0 more diverse, 0.1 more conservative")

        quick_gen_checkbox = gr.Checkbox(label="Quick Generation", value=True)
        settings_checkboxes = ["Use semantic history", "Use coarse history", "Use fine history", "Use last generation as history"]
        complete_settings = gr.CheckboxGroup(choices=settings_checkboxes, value=settings_checkboxes, label="Generation Settings", type="value", interactive=True, visible=False)
        quick_gen_checkbox.change(fn=on_quick_gen_changed, inputs=quick_gen_checkbox, outputs=complete_settings)

        output_audio = gr.Audio(label="Generated Audio", type="filepath")
        tts_create_button = gr.Button("Create")
    with gr.Tab("Clone Voice"):
        input_audio_filename = gr.Audio(label="Input audio.wav", source="upload", type="filepath")
        transcription_text = gr.Textbox(label="Transcription Text", lines=1, placeholder="Enter Text of your Audio Sample here...")
        initialname = "./bark/assets/prompts/MeMyselfAndI"
        #inputAudioFilename = gr.Textbox(label="Filename of Input Audio", lines=1, placeholder="audio.wav")
        output_voice = gr.Textbox(label="Filename of trained Voice", lines=1, placeholder=initialname, value=initialname)
        clone_voice_button = gr.Button("Create Voice")
        dummy = gr.Text(label="Progress")

        tts_create_button.click(generate_text_to_speech, inputs=[input_text, speaker, text_temp, waveform_temp, quick_gen_checkbox, complete_settings],outputs=output_audio)
        clone_voice_button.click(clone_voice, inputs=[input_audio_filename, transcription_text, output_voice], outputs=dummy)

barkgui.queue().launch(inbrowser=autolaunch)
