import os
import shutil


path_to_this_script = os.path.realpath(__file__)
dir_of_this_script = os.path.dirname(path_to_this_script)
saymore_dir = os.path.join(dir_of_this_script, "SayMoreFilesWKJ/")
sessions_dir = os.path.join(saymore_dir, "Sessions/")

sprj_fp, = [os.path.join(saymore_dir, fp) for fp in os.listdir(saymore_dir) if fp.endswith(".sprj")]  # will assert len 1 by unpacking
sprj_target = sprj_fp.replace("SayMoreFilesWKJ/", "")
print(f"copying\n\t{sprj_fp}\nto\n\t{sprj_target}\n")
shutil.copyfile(sprj_fp, sprj_target)

sessions_listdir = [os.path.join(sessions_dir, d) for d in os.listdir(sessions_dir)]  # why isn't there a kwarg for this
sessions_subdirs = [d for d in sessions_listdir if os.path.isdir(d)]
# print(sessions_subdirs)

for subdir in sessions_subdirs:
    fps = [os.path.join(subdir, fp) for fp in os.listdir(subdir)]
    # print("\n".join(fps))
    sound_extensions = [".wav", ".mp3", ".mp4"]
    not_sound = [fp for fp in fps if not any(fp.endswith(ext) for ext in sound_extensions)]
    for fp in not_sound:
        target = fp.replace("SayMoreFilesWKJ/", "")
        print(f"copying\n\t{fp}\nto\n\t{target}\n")
        shutil.copyfile(fp, target)

print("done")
