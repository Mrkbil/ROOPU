from faceswap_tab import on_srcfile_changed, on_use_face_from_selected, on_destfiles_changed, start_swap
import os


def get_files(directory, extension):
    if not os.path.isdir(directory):
        raise ValueError(f"The directory '{directory}' does not exist.")
    file_list = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            full_path = os.path.abspath(os.path.join(directory, filename))
            file_list.append(full_path)
    return file_list



upsample_options = ["128px", "256px", "512px"]
enhancer_options = ["None", "Codeformer", "DMDNet", "GFPGAN", "GPEN", "Restoreformer++"]
detection_options = ["Selected face", "First found", "All input faces","All faces", "All female", "All male"]
selected_mask_engine_options = ['Clip2Seg', 'DFL XSeg', None]
processing_method_options = ["Extract Frames to media", "In-Memory processing"]
no_face_action_options = ["Use untouched original frame", "Retry rotated", "Skip Frame",
                           "Skip Frame if no similar face", "Use last swapped"]


directory_path = "new/temp/homat/"
file_extension = ".png"
target_directory_path = "new/temp/homat/"


srcfiles=['E:\\Files\\ROOPU\\new\\m.jpg']
# target_faces=['E:\\Files\\ROOPU\\new\\s.png']
target_faces=[]
#target_faces=get_files(target_directory_path, file_extension)
destfiles=['E:\\Files\\ROOPU\\new\\homa.mp4']
# destfiles=get_files(directory_path, file_extension)

on_srcfile_changed(srcfiles)
on_use_face_from_selected(target_faces,1)
on_destfiles_changed(destfiles)
start_swap(
    upsample=upsample_options[0],  # ["128px", "256px", "512px"]
    enhancer=enhancer_options[0],  # ["None", "Codeformer", "DMDNet", "GFPGAN", "GPEN", "Restoreformer++"]
    detection=detection_options[0], # ["Selected face", "First found", "All input faces", "All faces", "All female", "All male"]
    keep_frames=True,
    wait_after_extraction=False,
    skip_audio=False,
    face_distance=0.95,
    blend_ratio=0.75,
    selected_mask_engine=selected_mask_engine_options[2],  # "DFL XSeg"
    clip_text="cup,hands,hair,banana",
    processing_method=processing_method_options[0],  # "Extract Frames to media"
    no_face_action=no_face_action_options[0],  # "Use untouched original frame"
    vr_mode=False,
    autorotate=False,
    restore_original_mouth=False,
    num_swap_steps=1,
    output_method='File'
)



