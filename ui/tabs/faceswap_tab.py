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



def faceswap_tab():
    return

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

def on_mask_engine_changed(mask_engine):
    print("on_mask_engine_changed")
    if mask_engine == "Clip2Seg":
        return gr.Textbox(interactive=True)
    return gr.Textbox(interactive=False)


def on_add_local_folder(folder):
    print("on_add_local_folder")
    files = util.get_local_files_from_folder(folder)
    if files is None:
        gr.Warning("Empty folder or folder not found!")
    return files


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


def on_select_input_face(evt: gr.SelectData):
    print("on_select_input_face")
    global SELECTED_INPUT_FACE_INDEX

    SELECTED_INPUT_FACE_INDEX = evt.index


def remove_selected_input_face():
    print("remove_selected_input_face")
    global SELECTED_INPUT_FACE_INDEX

    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        f = roop.globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)
        del f
    if len(ui.globals.ui_input_thumbs) > SELECTED_INPUT_FACE_INDEX:
        f = ui.globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
        del f

    return ui.globals.ui_input_thumbs

def move_selected_input(button_text):
    print("move_selected_input")
    global SELECTED_INPUT_FACE_INDEX

    if button_text == "⬅ Move left":
        if SELECTED_INPUT_FACE_INDEX <= 0:
            return ui.globals.ui_input_thumbs
        offset = -1
    else:
        if len(ui.globals.ui_input_thumbs) <= SELECTED_INPUT_FACE_INDEX:
            return ui.globals.ui_input_thumbs
        offset = 1
    
    f = roop.globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)
    roop.globals.INPUT_FACESETS.insert(SELECTED_INPUT_FACE_INDEX + offset, f)
    f = ui.globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
    ui.globals.ui_input_thumbs.insert(SELECTED_INPUT_FACE_INDEX + offset, f)
    return ui.globals.ui_input_thumbs
        

def move_selected_target(button_text):
    print("move_selected_target")
    global SELECTED_TARGET_FACE_INDEX

    if button_text == "⬅ Move left":
        if SELECTED_TARGET_FACE_INDEX <= 0:
            return ui.globals.ui_target_thumbs
        offset = -1
    else:
        if len(ui.globals.ui_target_thumbs) <= SELECTED_TARGET_FACE_INDEX:
            return ui.globals.ui_target_thumbs
        offset = 1
    
    f = roop.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
    roop.globals.TARGET_FACES.insert(SELECTED_TARGET_FACE_INDEX + offset, f)
    f = ui.globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
    ui.globals.ui_target_thumbs.insert(SELECTED_TARGET_FACE_INDEX + offset, f)
    return ui.globals.ui_target_thumbs




def on_select_target_face(evt: gr.SelectData):
    print("on_select_target_face")
    global SELECTED_TARGET_FACE_INDEX

    SELECTED_TARGET_FACE_INDEX = evt.index

def remove_selected_target_face():
    print("remove_selected_target_face")
    if len(ui.globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = roop.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    if len(ui.globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = ui.globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    return ui.globals.ui_target_thumbs


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
    return gr.Row(visible=False), None, ui.globals.ui_target_thumbs, gr.Dropdown(value='Selected face')

    #return gr.Row(visible=True), thumbs, gr.Gallery(visible=True), gr.Dropdown(visible=True)


    # roop.globals.target_path = files[0]
    #
    # if util.is_image(roop.globals.target_path) and not roop.globals.target_path.lower().endswith(('gif')):
    #     SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path,  (False, 0))
    #     if len(SELECTION_FACES_DATA) > 0:
    #         for f in SELECTION_FACES_DATA:
    #             image = util.convert_to_gradio(f[1])
    #             thumbs.append(image)
    #     else:
    #         gr.Info('No faces detected!')
    #         roop.globals.target_path = None
    #
    # elif util.is_video(roop.globals.target_path) or roop.globals.target_path.lower().endswith(('gif')):
    #     selected_frame = frame_num
    #     SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path, (True, selected_frame))
    #     if len(SELECTION_FACES_DATA) > 0:
    #         for f in SELECTION_FACES_DATA:
    #             image = util.convert_to_gradio(f[1])
    #             thumbs.append(image)
    #     else:
    #         gr.Info('No faces detected!')
    #         roop.globals.target_path = None
    # else:
    #     gr.Info('Unknown image/video type!')
    #     roop.globals.target_path = None

    # if len(thumbs) == 1:
    #     roop.globals.TARGET_FACES.append(SELECTION_FACES_DATA[0][0])
    #     ui.globals.ui_target_thumbs.append(thumbs[0])
    #     print("x",len(roop.globals.TARGET_FACES) )
    #     return gr.Row(visible=False), None, ui.globals.ui_target_thumbs, gr.Dropdown(value='Selected face')
    #
    # return gr.Row(visible=True), thumbs, gr.Gallery(visible=True), gr.Dropdown(visible=True)


def on_select_face(evt: gr.SelectData):  # SelectData is a subclass of EventData
    print("on_select_face")
    global SELECTED_FACE_INDEX
    SELECTED_FACE_INDEX = evt.index


def on_selected_face():
    print("on_selected_face")
    global IS_INPUT, SELECTED_FACE_INDEX, SELECTION_FACES_DATA
    
    fd = SELECTION_FACES_DATA[SELECTED_FACE_INDEX]
    image = util.convert_to_gradio(fd[1])
    if IS_INPUT:
        face_set = FaceSet()
        fd[0].mask_offsets = (0,0,0,0,1,20)
        face_set.faces.append(fd[0])
        roop.globals.INPUT_FACESETS.append(face_set)
        ui.globals.ui_input_thumbs.append(image)
        return ui.globals.ui_input_thumbs, gr.Gallery(visible=True), gr.Dropdown(visible=True)
    else:
        roop.globals.TARGET_FACES.append(fd[0])
        print(roop.globals.TARGET_FACES)
        ui.globals.ui_target_thumbs.append(image)
        return gr.Gallery(visible=True), ui.globals.ui_target_thumbs, gr.Dropdown(value='Selected face')

#        bt_faceselect.click(fn=on_selected_face, outputs=[dynamic_face_selection, face_selection, input_faces, target_faces])

def on_end_face_selection():
    print("on_end_face_selection")
    return gr.Column(visible=False), None


def save_masked_image(maskimage):
    # Convert NumPy array (maskimage) to PIL Image in RGBA mode
    pil_image = Image.fromarray(maskimage.astype(np.uint8))

    # Convert RGBA to RGB (remove the alpha channel)
    pil_image_rgb = pil_image.convert("RGB")

    # Save the image as a .jpg file
    pil_image_rgb.save("masked_image.jpg", format="JPEG")
    return "masked_image.jpg"
# Preview frame
def on_preview_frame_changed(frame_num, files, fake_preview, enhancer, detection, face_distance, blend_ratio,
                              selected_mask_engine, clip_text, no_face_action, vr_mode, auto_rotate, maskimage, show_face_area, restore_original_mouth, num_steps, upsample):
    print("on_preview_frame_changed")
    # print(f"""
    # frame_num: {frame_num}
    # files: {files}
    # fake_preview: {fake_preview}
    # enhancer: {enhancer}
    # detection: {detection}
    # face_distance: {face_distance}
    # blend_ratio: {blend_ratio}
    # selected_mask_engine: {selected_mask_engine}
    # clip_text: {clip_text}
    # no_face_action: {no_face_action}
    # vr_mode: {vr_mode}
    # auto_rotate: {auto_rotate}
    #
    # show_face_area: {show_face_area}
    # restore_original_mouth: {restore_original_mouth}
    # num_steps: {num_steps}
    # upsample: {upsample}
    # """)

    global SELECTED_INPUT_FACE_INDEX, manual_masking, current_video_fps

    from roop.core import live_swap, get_processing_plugins

    manual_masking = False
    mask_offsets = (0,0,0,0)
    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        if not hasattr(roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0], 'mask_offsets'):
            roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets = mask_offsets
        mask_offsets = roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets

    timeinfo = '0:00:00'
    if files is None or selected_preview_index >= len(files) or frame_num is None:
        return None,None, gr.Slider(info=timeinfo)

    filename = files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, frame_num)
        if current_video_fps == 0:
            current_video_fps = 1
        secs = (frame_num - 1) / current_video_fps
        minutes = secs / 60
        secs = secs % 60
        hours = minutes / 60
        minutes = minutes % 60
        milliseconds = (secs - int(secs)) * 1000
        timeinfo = f"{int(hours):0>2}:{int(minutes):0>2}:{int(secs):0>2}.{int(milliseconds):0>3}"  
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None:
        return None, None, gr.Slider(info=timeinfo)
    
    # layers = None
    # if maskimage is not None:
    #     layers = maskimage["layers"]

    if not fake_preview or len(roop.globals.INPUT_FACESETS) < 1:
        return gr.Image(value=util.convert_to_gradio(current_frame), visible=True), gr.ImageEditor(visible=False), gr.Slider(info=timeinfo)

    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.selected_enhancer = enhancer
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.no_face_action = index_of_no_face_action(no_face_action)
    roop.globals.vr_mode = vr_mode
    roop.globals.autorotate_faces = auto_rotate
    roop.globals.subsample_size = int(upsample[:3])


    mask_engine = map_mask_engine(selected_mask_engine, clip_text)

    roop.globals.execution_threads = roop.globals.CFG.max_threads
    #mask = layers[0] if layers is not None else None
    face_index = SELECTED_INPUT_FACE_INDEX
    if len(roop.globals.INPUT_FACESETS) <= face_index:
        face_index = 0
   
    options = ProcessOptions(get_processing_plugins(mask_engine), roop.globals.distance_threshold, roop.globals.blend_ratio,
                              roop.globals.face_swap_mode, face_index, clip_text, maskimage, num_steps, roop.globals.subsample_size, show_face_area, restore_original_mouth)

    current_frame = live_swap(current_frame, options)
    if current_frame is None:
        return gr.Image(visible=True), None, gr.Slider(info=timeinfo)
    return gr.Image(value=util.convert_to_gradio(current_frame), visible=True), gr.ImageEditor(visible=False), gr.Slider(info=timeinfo)

def map_mask_engine(selected_mask_engine, clip_text):
    print("map_mask_engine")
    if selected_mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
        if clip_text is None or len(clip_text) < 1:
          mask_engine = None
    elif selected_mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    else:
        mask_engine = None
    return mask_engine


def on_toggle_masking(previewimage, mask):
    print("on_toggle_masking")
    global manual_masking

    manual_masking = not manual_masking
    if manual_masking:
        layers = mask["layers"]
        if len(layers) == 1:
            layers = [create_blank_image(previewimage.shape[1],previewimage.shape[0])]
        return gr.Image(visible=False), gr.ImageEditor(value={"background": previewimage, "layers": layers, "composite": None}, visible=True)
    return gr.Image(visible=True), gr.ImageEditor(visible=False)

def gen_processing_text(start, end):
    print("gen_processing_text")
    return f'Processing frame range [{start} - {end}]'

def on_set_frame(sender:str, frame_num):
    print("on_set_frame")
    global selected_preview_index, list_files_process
    
    idx = selected_preview_index
    if list_files_process[idx].endframe == 0:
        return gen_processing_text(0,0)
    
    start = list_files_process[idx].startframe
    end = list_files_process[idx].endframe
    if sender.lower().endswith('start'):
        list_files_process[idx].startframe = min(frame_num, end)
    else:
        list_files_process[idx].endframe = max(frame_num, start)
    
    return gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe)


def on_preview_mask(frame_num, files, clip_text, mask_engine):
    print("on_preview_mask")
    from roop.core import live_swap, get_processing_plugins
    global is_processing

    if is_processing or files is None or selected_preview_index >= len(files) or clip_text is None or frame_num is None:
        return None
        
    filename = files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, frame_num
                                        )
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None or mask_engine is None:
        return None
    if mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
        if clip_text is None or len(clip_text) < 1:
          mask_engine = None
    elif mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    options = ProcessOptions(get_processing_plugins(mask_engine), roop.globals.distance_threshold, roop.globals.blend_ratio,
                              "all", 0, clip_text, None, 0, 128, False, False, True)

    current_frame = live_swap(current_frame, options)
    return util.convert_to_gradio(current_frame)


def on_clear_input_faces():
    print("on_clear_input_faces")
    ui.globals.ui_input_thumbs.clear()
    roop.globals.INPUT_FACESETS.clear()
    return ui.globals.ui_input_thumbs

def on_clear_destfiles():
    print("on_clear_destfiles")
    roop.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs.clear()
    return ui.globals.ui_target_thumbs, gr.Dropdown(value="First found")    


def index_of_no_face_action(dropdown_text):
    print("index_of_no_face_action")
    global no_face_choices

    return no_face_choices.index(dropdown_text) 

def translate_swap_mode(dropdown_text):
    print("translate_swap_mode")
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





def stop_swap():
    print("stop_swap")
    roop.globals.processing = False
    gr.Info('Aborting processing - please wait for the remaining threads to be stopped')
    return gr.Button(variant="primary", interactive=True),gr.Button(variant="secondary", interactive=False),None


def on_fps_changed(fps):
    print("on_fps_changed")
    global selected_preview_index, list_files_process

    if len(list_files_process) < 1 or list_files_process[selected_preview_index].endframe < 1:
        return
    list_files_process[selected_preview_index].fps = fps


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
    
    if total_frames > 1:
        return gr.Slider(value=list_files_process[idx].startframe, maximum=total_frames, info='0:00:00'), gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe), fps
    return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), gen_processing_text(0,0), fps


def on_resultfiles_selected(evt: gr.SelectData, files):
    print("on_resultfiles_selected")
    selected_index = evt.index
    filename = files[selected_index].name
    return display_output(filename)

def on_resultfiles_finished(files):
    print("on_resultfiles_finished")
    selected_index = 0
    if files is None or len(files) < 1:
        return None, None
    
    filename = files[selected_index].name
    return display_output(filename)


def get_gradio_output_format():
    print("get_gradio_output_format")
    if roop.globals.CFG.output_image_format == "jpg":
        return "jpeg"
    return roop.globals.CFG.output_image_format


def display_output(filename):
    print("display_output")
    if util.is_video(filename) and roop.globals.CFG.output_show_video:
        return gr.Image(visible=False), gr.Video(visible=True, value=filename)
    else:
        if util.is_video(filename) or filename.lower().endswith('gif'):
            current_frame = get_video_frame(filename)
        else:
            current_frame = get_image_frame(filename)
        return gr.Image(visible=True, value=util.convert_to_gradio(current_frame)), gr.Video(visible=False)


def start_swap(progress=gr.Progress()):
    print("A")
    imagemask = None
    print("start_swap")
    output_method = "File"
    enhancer = "None"  # GFPGAN
    detection = "First found"
    keep_frames = False
    wait_after_extraction = False
    skip_audio = False
    face_distance = 0.65
    blend_ratio = 0.65
    selected_mask_engine = "None"
    clip_text = "cup,hands,hair,banana"
    processing_method = "In-Memory processing"
    no_face_action = "Use untouched original frame"
    vr_mode = False
    autorotate = True
    restore_original_mouth = False
    num_swap_steps = 1
    upsample = "128px"

    # # print("Image Mask:", imagemask)
    #
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
    print("batch aq")
    if roop.globals.face_swap_mode == 'selected':
        if len(roop.globals.TARGET_FACES) < 1:
            print("TARGET_FACES None")
            # gr.Error('No Target Face selected!')
            # return gr.Button(variant="primary"), None, None
    print("batch a")
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



srcfiles=['E:\\Files\\ROOPU\\new\\Rakibul_Islam_sq.jpg']
target=['E:\\Files\\ROOPU\\new\\trgt.jpg']
destfiles=['E:\\Files\\ROOPU\\new\\trgt.jpg']


on_srcfile_changed(srcfiles)
on_use_face_from_selected(target,1)
on_destfiles_changed(destfiles)
start_swap()