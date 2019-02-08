from corral.dataset import AIDataSet
from pathlib import Path

path_sample_root = Path.cwd() / 'sample'
assert path_sample_root.exists()
path_data = Path.cwd() / 'sample'
data_folder = '20181104 1200'

AIDataSet(path_data, data_folder)
def test_load():
    # ds = AIDataSet(LOCAL_PROJECT_PATH, DATASET_ID)
    path_jpg = path_sample_root / 'jpg'
    path_json = path_sample_root / 'json'
    path_jpg.glob('*.jpg')
    assert True
