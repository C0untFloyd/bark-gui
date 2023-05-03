from cProfile import label
from distutils.command.check import check
from doctest import Example
import gradio as gr
import os
import sys
import numpy as np
import logging
import torch
from xml.sax import saxutils
#import nltk

from bark import SAMPLE_RATE, generate_audio, set_seed
from bark.clonevoice import clone_voice
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
from scipy.io.wavfile import write as write_wav
from parseinput import split_and_recombine_text, build_ssml, is_ssml, create_clips_from_ssml
from datetime import datetime
from tqdm.auto import tqdm

OUTPUTFOLDER = "Outputs"


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

def generate_text_to_speech(text, selected_speaker, text_temp, waveform_temp, quick_generation, complete_settings, random_seed_number, progress=gr.Progress(track_tqdm=True)):
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
    random_seed = int(random_seed_number)

    progress(0, desc="Generating")

    silenceshort = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)  # quarter second of silence
    silencelong = np.zeros(int(0.50 * SAMPLE_RATE), dtype=np.float32)  # half a second of silence

    all_parts = []
    text = text.lstrip()
    if is_ssml(text):
        list_speak = create_clips_from_ssml(text)
        prev_speaker = None
        for i, clip in tqdm(enumerate(list_speak), total=len(list_speak)):
            # set seed for consistent generation
            set_seed(random_seed)
            selected_speaker = clip[0]
            # Add pause break between speakers
            if i > 0 and selected_speaker != prev_speaker:
                all_parts += [silencelong.copy()]
            prev_speaker = selected_speaker
            text = clip[1]
            text = saxutils.unescape(text)
            if selected_speaker == "None":
                selected_speaker = None

            print(f"\nGenerating Text ({i+1}/{len(list_speak)}) -> {selected_speaker}:`{text}`")
            audio_array = generate_audio(text, selected_speaker, text_temp, waveform_temp)
            if len(list_speak) > 1:
                save_wav(audio_array, create_filename(OUTPUTFOLDER, "audioclip",".wav"))
            all_parts += [audio_array]
    else:
        texts = split_and_recombine_text(text)
        for i, text in tqdm(enumerate(texts), total=len(texts)):
            # set seed for consistent generation
            set_seed(random_seed)
            print(f"\nGenerating Text ({i+1}/{len(texts)}) -> {selected_speaker}:`{text}`")
            if quick_generation == True:
                audio_array = generate_audio(text, selected_speaker, text_temp, waveform_temp)

            else:
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

            # Noticed this in the HF Demo - convert to 16bit int -32767/32767 - most used audio format  
            # audio_array = (audio_array * 32767).astype(np.int16)

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
            all_parts += [audio_array]
            # Add short pause between sentences
            if text[-1] in "!?.\n" and i > 1:
                all_parts += [silenceshort.copy()]

            #all_parts += [audio_array, silencelong.copy()]
    # reset seed
    set_seed(-1)
    
    # save & play audio
    result = create_filename(OUTPUTFOLDER, "final",".wav")
    save_wav(np.concatenate(all_parts), result)
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

def delete_output_files(checkbox_state):
    if checkbox_state:
        outputs_folder = os.path.join(os.getcwd(), OUTPUTFOLDER)
        if os.path.exists(outputs_folder):
            purgedir(outputs_folder)
    return False


# https://stackoverflow.com/a/54494779
def purgedir(parent):
    for root, dirs, files in os.walk(parent):                                      
        for item in files:
            # Delete subordinate files                                                 
            filespec = os.path.join(root, item)
            os.unlink(filespec)
        for item in dirs:
            # Recursively perform this operation for subordinate directories   
            purgedir(os.path.join(root, item))

def convert_text_to_ssml(text, selected_speaker):
    return build_ssml(text, selected_speaker)

    

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
print(f'autolaunch={autolaunch}\n\n')

#print("Updating nltk\n")
#nltk.download('punkt')

print("Preloading Models\n")
preload_models()

# Collect all existing speakers/voices in dir
speakers_list = []

for root, dirs, files in os.walk("./bark/assets/prompts"):
	for file in files:
		if(file.endswith(".npz")):
			pathpart = root.replace("./bark/assets/prompts", "")
			name = os.path.join(pathpart, file[:-4])
			if name.startswith("/") or name.startswith("\\"):
				name = name[1:]
			speakers_list.append(name)

speakers_list = sorted(speakers_list, key=lambda x: x.lower())
speakers_list.insert(0, 'None')

# Create Gradio Blocks

with gr.Blocks(title="Bark Enhanced Gradio GUI", mode="Bark Enhanced") as barkgui:
    gr.Markdown("### [Bark Enhanced v0.4.0](https://github.com/C0untFloyd/bark-gui)")
    with gr.Tab("TTS"):
        with gr.Row():
            with gr.Column():
                placeholder = "Enter text here."
                input_text = gr.Textbox(label="Input Text", lines=4, placeholder=placeholder)
            with gr.Column():
                convert_to_ssml_button = gr.Button("Convert Text to SSML")
        with gr.Row():
            with gr.Column():
                examples = [
                    "Special meanings: [laughter] [laughs] [sighs] [music] [gasps] [clears throat] MAN: WOMAN:",
                   "♪ Never gonna make you cry, never gonna say goodbye, never gonna tell a lie and hurt you ♪",
                   "And now — a picture of a larch [laughter]",
                   """
                        WOMAN: I would like an oatmilk latte please.
                        MAN: Wow, that's expensive!
                   """,
                   """<?xml version="1.0"?>
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://www.w3.org/2001/10/synthesis
                   http://www.w3.org/TR/speech-synthesis/synthesis.xsd"
         xml:lang="en-US">
<voice name="/v2/en_speaker_9">Look at that drunk guy!</voice>
<voice name="/v2/en_speaker_3">Who is he?</voice>
<voice name="/v2/en_speaker_9">WOMAN: [clears throat] 10 years ago, he proposed me and I rejected him.</voice>
<voice name="/v2/en_speaker_3">Oh my God [laughs] he is still celebrating</voice>
</speak>"""
                   ]
                examples = gr.Examples(examples=examples, inputs=input_text)

        with gr.Row():
            with gr.Column():
                gr.Markdown("[Voice Prompt Library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)")
                speaker = gr.Dropdown(speakers_list, value=speakers_list[0], label="Voice")
            with gr.Column():
                text_temp = gr.Slider(0.1, 1.0, value=0.6, label="Generation Temperature", info="1.0 more diverse, 0.1 more conservative")
                waveform_temp = gr.Slider(0.1, 1.0, value=0.7, label="Waveform temperature", info="1.0 more diverse, 0.1 more conservative")

        with gr.Row():
            with gr.Column():
                quick_gen_checkbox = gr.Checkbox(label="Quick Generation", value=True)
            with gr.Column():
                with gr.Row():
                    settings_checkboxes = ["Use semantic history", "Use coarse history", "Use fine history", "Use last generation as history"]
                    complete_settings = gr.CheckboxGroup(choices=settings_checkboxes, value=settings_checkboxes, label="Detailed Generation Settings", type="value", interactive=True, visible=False)
                    random_seed_number = gr.inputs.Number(label="Random Seed", default=-1)
                    #random_seed_settings = gr.NumberGroup([random_seed_number], label="Random Seed Settings", interactive=True, visible=False)
                    quick_gen_checkbox.change(fn=on_quick_gen_changed, inputs=quick_gen_checkbox, outputs=[complete_settings])

        with gr.Row():
            with gr.Column():
                tts_create_button = gr.Button("Generate")
            with gr.Column():
                hidden_checkbox = gr.Checkbox(visible=False)
                button_delete_files = gr.Button("Clear output folder")
        with gr.Row():
            output_audio = gr.Audio(label="Generated Audio", type="filepath")

    with gr.Tab("Clone Voice"):
        input_audio_filename = gr.Audio(label="Input audio.wav", source="upload", type="filepath")
        transcription_text = gr.Textbox(label="Transcription Text", lines=1, placeholder="Enter Text of your Audio Sample here...")
        initialname = "./bark/assets/prompts/custom/MeMyselfAndI"
        #inputAudioFilename = gr.Textbox(label="Filename of Input Audio", lines=1, placeholder="audio.wav")
        output_voice = gr.Textbox(label="Filename of trained Voice", lines=1, placeholder=initialname, value=initialname)
        clone_voice_button = gr.Button("Create Voice")
        dummy = gr.Text(label="Progress")

        convert_to_ssml_button.click(convert_text_to_ssml, inputs=[input_text, speaker],outputs=input_text)
        tts_create_button.click(generate_text_to_speech, inputs=[input_text, speaker, text_temp, waveform_temp, quick_gen_checkbox, complete_settings, random_seed_number],outputs=output_audio)
        # Javascript hack to display modal confirmation dialog
        js = "(x) => confirm('Are you sure? This will remove all files from output folder')"
        button_delete_files.click(None, None, hidden_checkbox, _js=js)
        hidden_checkbox.change(delete_output_files, [hidden_checkbox], [hidden_checkbox])
        clone_voice_button.click(clone_voice, inputs=[input_audio_filename, transcription_text, output_voice], outputs=dummy)

barkgui.queue().launch(inbrowser=autolaunch)
