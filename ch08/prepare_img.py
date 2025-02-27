import os, shutil, pathlib

script_dir = pathlib.Path(os.path.dirname(__file__))
original_dir = script_dir / "../../PetImages"
new_base_dir = script_dir / "cats_vs_dogs_small"
      
def make_subset(subset_name, start_index, end_index):
    for (category_src, category_dst) in [("Cat", "cat"), ("Dog", "dog")]:
        dir = new_base_dir / subset_name / category_dst
        os.makedirs(dir, exist_ok=True)
        fnames_src = [f"{i}.jpg" for i in range(start_index, end_index)]
        fnames_dst = [f"{category_dst}.{i}.jpg" for i in range(start_index, end_index)]
        for i, (fname_src, fname_dst) in enumerate(zip(fnames_src, fnames_dst)):
            try:
                shutil.copyfile(src=original_dir / category_src / fname_src, dst=dir / fname_dst)
            except FileNotFoundError:
                print(f"Copy {fnames_dst[i-1]} to {fnames_dst[i]}")
                shutil.copyfile(src=dir / fnames_dst[i-1], dst=dir / fname_dst)

make_subset("train", start_index=0, end_index=1000)
make_subset("validation", start_index=1000, end_index=1500)
make_subset("test", start_index=1500, end_index=2500)