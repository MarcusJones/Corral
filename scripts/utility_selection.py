#%% imports
from pathlib import Path
import zipfile
import shutil
#%%
PATH_OUT = Path("/home/batman/MULE DATA/out")

#%%
path_jpg = Path("/home/batman/MULE DATA/20180907 184100 BENCHMARK1 TRG/jpg_images")
jpgs=list()
for i,file_path in enumerate(sorted(path_jpg.glob('*.jpg'))):
    if i>100 and i<201:
        out_path = PATH_OUT / file_path.name
        jpgs.append(file_path)
        shutil.copyfile(file_path,out_path)
    # file_path.name
    # file_path.suffix
    # file_path.stem
#%%
path_json = Path("/home/batman/MULE DATA/20180907 184100 BENCHMARK1 TRG/json/json_records")
jpgs=list()
for i,file_path in enumerate(sorted(path_json.glob('*.json'))):
    if i>100 and i<201:
        out_path = PATH_OUT / file_path.name
        jpgs.append(file_path)
        shutil.copyfile(file_path,out_path)
    # file_path.name
    # file_path.suffix
    # file_path.stem
#%%
path_zip = Path("/home/batman/MULE DATA/20180907 184100 BENCHMARK1 TRG/json_records.zip")
json_records = list()
with zipfile.ZipFile(path_zip, "r") as f:
    json_file_paths = [name for name in f.namelist() if os.path.splitext(name)[1] == '.json']
    # Each record is a seperate json file
    for json_file in json_file_paths:
        # this_fname = os.path.splitext(json_file)[0]
        this_timestep = this_fname.split('_')[1]
        d = f.read(json_file)
        d = json.loads(d.decode("utf-8"))
        d['timestamp'] = this_timestep
        json_records.append(d)