import os
import shutil
import pathlib
import gradio as gr
import numpy as np
from PIL import Image
from settings import Settings
import roop.utilities as util
import roop.globals
import ui.globals
from roop.face_util import extract_face_images, create_blank_image
from roop.capturer import get_video_frame, get_video_frame_total, get_image_frame
from roop.ProcessEntry import ProcessEntry
from roop.ProcessOptions import ProcessOptions
from roop.FaceSet import FaceSet
from roop.utilities import clean_dir

last_image = None


IS_INPUT = True
SELECTED_FACE_INDEX = 0

SELECTED_INPUT_FACE_INDEX = 0
SELECTED_TARGET_FACE_INDEX = 0

input_faces = None
target_faces = None
face_selection = None
previewimage = None

selected_preview_index = 0

is_processing = False            

list_files_process : list[ProcessEntry] = []
no_face_choices = ["Use untouched original frame","Retry rotated", "Skip Frame", "Skip Frame if no similar face", "Use last swapped"]
swap_choices = ["First found", "All input faces", "All female", "All male", "All faces", "Selected face"]

current_video_fps = 50

manual_masking = False




def on_mask_top_changed(mask_offset):
    set_mask_offset(0, mask_offset)

def on_mask_bottom_changed(mask_offset):
    set_mask_offset(1, mask_offset)

def on_mask_left_changed(mask_offset):
    set_mask_offset(2, mask_offset)

def on_mask_right_changed(mask_offset):
    set_mask_offset(3, mask_offset)

def on_mask_erosion_changed(mask_offset):
    set_mask_offset(4, mask_offset)
def on_mask_blur_changed(mask_offset):
    set_mask_offset(5, mask_offset)


def set_mask_offset(index, mask_offset):
    print("set_mask_offset")
    global SELECTED_INPUT_FACE_INDEX

    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        offs = roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets
        offs[index] = mask_offset
        if offs[0] + offs[1] > 0.99:
            offs[0] = 0.99
            offs[1] = 0.0
        if offs[2] + offs[3] > 0.99:
            offs[2] = 0.99
            offs[3] = 0.0
        roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets = offs


def translate_swap_mode(dropdown_text):
    if dropdown_text == "Selected face":
        return "selected"
    elif dropdown_text == "First found":
        return "first"
    elif dropdown_text == "All input faces":
        return "all_input"
    elif dropdown_text == "All female":
        return "all_female"
    elif dropdown_text == "All male":
        return "all_male"

    return "all"


def index_of_no_face_action(dropdown_text):
    global no_face_choices

    return no_face_choices.index(dropdown_text)

def map_mask_engine(selected_mask_engine, clip_text):
    if selected_mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
        if clip_text is None or len(clip_text) < 1:
          mask_engine = None
    elif selected_mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    else:
        mask_engine = None
    return mask_engine

def on_srcfile_changed(srcfiles, progress=gr.Progress()):

    global SELECTION_FACES_DATA, IS_INPUT, input_faces, face_selection, last_image
    print(srcfiles)
    IS_INPUT = True

    if srcfiles is None or len(srcfiles) < 1:
        return gr.Column(visible=False), None, ui.globals.ui_input_thumbs, None

    for f in srcfiles:    
        #source_path = f.name
        source_path = f
        if source_path.lower().endswith('fsz'):
            #progress(0, desc="Retrieving faces from Faceset File")
            unzipfolder = os.path.join(os.environ["TEMP"], 'faceset')
            if os.path.isdir(unzipfolder):
                files = os.listdir(unzipfolder)
                for file in files:
                    os.remove(os.path.join(unzipfolder, file))
            else:
                os.makedirs(unzipfolder)
            util.mkdir_with_umask(unzipfolder)
            util.unzip(source_path, unzipfolder)
            is_first = True
            face_set = FaceSet()
            for file in os.listdir(unzipfolder):
                if file.endswith(".png"):
                    filename = os.path.join(unzipfolder,file)
                    #progress(0, desc="Extracting faceset")
                    SELECTION_FACES_DATA = extract_face_images(filename,  (False, 0))
                    for f in SELECTION_FACES_DATA:
                        face = f[0]
                        face.mask_offsets = (0,0,0,0,1,20)
                        face_set.faces.append(face)
                        if is_first: 
                            # image = util.convert_to_gradio(f[1])
                            # ui.globals.ui_input_thumbs.append(image)
                            is_first = False
                        face_set.ref_images.append(get_image_frame(filename))
            if len(face_set.faces) > 0:
                if len(face_set.faces) > 1:
                    face_set.AverageEmbeddings()
                roop.globals.INPUT_FACESETS.append(face_set)
                                        
        elif util.has_image_extension(source_path):
            #progress(0, desc="Retrieving faces from image")
            roop.globals.source_path = source_path

            SELECTION_FACES_DATA = extract_face_images(roop.globals.source_path,  (False, 0))
            #progress(0.5, desc="Retrieving faces from image")
            for f in SELECTION_FACES_DATA:
                face_set = FaceSet()
                face = f[0]
                face.mask_offsets = (0,0,0,0,1,20)
                face_set.faces.append(face)
                # image = util.convert_to_gradio(f[1])
                # ui.globals.ui_input_thumbs.append(image)
                roop.globals.INPUT_FACESETS.append(face_set)

    print(roop.globals.INPUT_FACESETS)
    #progress(1.0)
    return None, None, None,None






def on_use_face_from_selected(files, frame_num):

    print(files,frame_num)
    print("on_use_face_from_selected -set target face")
    global IS_INPUT, SELECTION_FACES_DATA
    IS_INPUT = False
    thumbs = []
    
    #roop.globals.target_path = files[selected_preview_index].name
    for img_path in files:
        roop.globals.target_path=img_path
        if util.is_image(roop.globals.target_path) and not roop.globals.target_path.lower().endswith(('gif')):
            SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path, (False, 0))
            if len(SELECTION_FACES_DATA) > 0:
                for f in SELECTION_FACES_DATA:
                    image = util.convert_to_gradio(f[1])
                    thumbs.append(image)
            else:
                gr.Info('No faces detected!')
                roop.globals.target_path = None
        elif util.is_video(roop.globals.target_path) or roop.globals.target_path.lower().endswith(('gif')):
            selected_frame = frame_num
            SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path, (True, selected_frame))
            if len(SELECTION_FACES_DATA) > 0:
                for f in SELECTION_FACES_DATA:
                    image = util.convert_to_gradio(f[1])
                    thumbs.append(image)
            else:
                gr.Info('No faces detected!')
                roop.globals.target_path = None
        else:
            gr.Info('Unknown image/video type!')
            roop.globals.target_path = None

        roop.globals.TARGET_FACES.append(SELECTION_FACES_DATA[0][0])
    for i in thumbs:
        ui.globals.ui_target_thumbs.append(i)
    print("x", len(roop.globals.TARGET_FACES))



def on_destfiles_changed(destfiles):
    print("on_destfiles_changed - Added file to process list") #

    global selected_preview_index, list_files_process, current_video_fps
    #print(selected_preview_index,list_files_process,current_video_fps) # 0 [] 50

    if destfiles is None or len(destfiles) < 1:
        list_files_process.clear()
        return
        #return gr.Slider(value=1, maximum=1, info='0:00:00'), ''
    
    for f in destfiles:
        #print(f)
        list_files_process.append(ProcessEntry(f, 0,0, 0))
    # selected_preview_index = 0
    # idx = selected_preview_index
    # #print("idx",idx)
    # filename = list_files_process[idx].filename
    # #print("filename ",filename)
    # if util.is_video(filename) or filename.lower().endswith('gif'):
    #     total_frames = get_video_frame_total(filename)
    #     if total_frames is None or total_frames < 1:
    #         total_frames = 1
    #         gr.Warning(f"Corrupted video {filename}, can't detect number of frames!")
    #     else:
    #         current_video_fps = util.detect_fps(filename)
    # else:
    #     total_frames = 1
    # list_files_process[idx].endframe = total_frames
    # print(list_files_process[idx].filename)
    # print(list_files_process[idx].endframe)
    # print("fps:",list_files_process[idx].fps)
    print("len:::::: ",len(list_files_process))
    # if total_frames > 1:
    #     return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe)
    # return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), ''


def on_destfiles_selected(evt: gr.SelectData):
    print("on_destfiles_selected")
    global selected_preview_index, list_files_process, current_video_fps

    if evt is not None:
        selected_preview_index = evt.index
    idx = selected_preview_index    
    filename = list_files_process[idx].filename
    fps = list_files_process[idx].fps
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
        current_video_fps = util.detect_fps(filename)
        if list_files_process[idx].endframe == 0:
            list_files_process[idx].endframe = total_frames 
    else:
        total_frames = 1
    
    # if total_frames > 1:
    #     return gr.Slider(value=list_files_process[idx].startframe, maximum=total_frames, info='0:00:00'), gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe), fps
    # return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), gen_processing_text(0,0), fps



def start_swap(upsample, enhancer, detection, keep_frames, wait_after_extraction,
               skip_audio, face_distance, blend_ratio, selected_mask_engine,
               clip_text, processing_method, no_face_action, vr_mode, autorotate,
               restore_original_mouth, num_swap_steps,output_method,progress=gr.Progress()):
    print("start_swap")
    imagemask = None
    # print("Image Mask:", imagemask)
    from ui.main import prepare_environment
    from roop.core import batch_process_regular
    global is_processing, list_files_process
    print("isprocessing ", is_processing)
    print("list_files_process ", list_files_process)
    # for p in list_files_process:
    #     print(p.filename,p.finalname)
    #     print(p.fps)
    #     print(p.startframe,p.endframe)

    if list_files_process is None or len(list_files_process) <= 0:
        return gr.Button(variant="primary"), None, None
    #

    if False:#roop.globals.CFG.clear_output:
        clean_dir(roop.globals.output_path)

    print("batch a3")
    if not util.is_installed("ffmpeg"):
        msg = "ffmpeg is not installed! No video processing possible."
        gr.Warning(msg)

    roop.globals.CFG = Settings(config_file="config.yaml")
    prepare_environment()
    #
    roop.globals.selected_enhancer = enhancer
    roop.globals.target_path = None
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.keep_frames = keep_frames
    roop.globals.wait_after_extraction = wait_after_extraction
    roop.globals.skip_audio = skip_audio
    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.no_face_action = index_of_no_face_action(no_face_action)
    roop.globals.vr_mode = vr_mode
    roop.globals.autorotate_faces = autorotate
    roop.globals.subsample_size = int(upsample[:3])
    mask_engine = map_mask_engine(selected_mask_engine, clip_text)

    if roop.globals.face_swap_mode == 'selected':
        if len(roop.globals.TARGET_FACES) < 1:
            print("TARGET_FACES None")
            # gr.Error('No Target Face selected!')
            # return gr.Button(variant="primary"), None, None

    is_processing = True
    #yield gr.Button(variant="secondary", interactive=False), gr.Button(variant="primary", interactive=True), None

    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    print("batch strat")
    batch_process_regular(output_method, list_files_process, mask_engine, clip_text,
                          processing_method == "In-Memory processing", imagemask, restore_original_mouth,
                          num_swap_steps, progress, SELECTED_INPUT_FACE_INDEX)
    print("batch end")



