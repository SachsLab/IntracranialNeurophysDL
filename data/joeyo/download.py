import csv
import math
import hashlib
import requests
import tqdm
from pathlib import Path


def get_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_from_web(remote_url, local_path, md5):
    if local_path.exists():
        if get_md5(local_path) == md5:
            print("{} already exists and md5 checks out. Skipping.".format(local_path))
            return
        else:
            print("{} already exists but md5 is bad. Replacing.".format(local_path))
            local_path.unlink()
    r = requests.get(remote_url, stream=True)  # stream=True does not download yet.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    with open(local_path, 'wb') as f:
        for data in tqdm.tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size),
                              unit='KB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR downloading to {}. Deleting.".format(local_path))
        local_path.unlink()
    elif get_md5(local_path) != md5:
        print("ERROR md5 mismatch for {}. Deleting.".format(local_path))
        local_path.unlink()
    print("Finished downloading {}".format(local_path))


if __name__ == "__main__":
    working_dir = Path.cwd() / 'data' / 'joeyo'
    datasets_file = working_dir / 'datasets.csv'
    # Get the list of datasets to download
    datasets = []
    with open(datasets_file) as csvfile:
        datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in datasetreader:
            datasets.append(row)

    # Create a local folder to store the data
    local_dir = working_dir / 'download'
    if not local_dir.is_dir():
        local_dir.mkdir()

    base_url = 'https://zenodo.org/record/'
    file_grp = '583331'
    for row in datasets:
        _fname = row['filename'] + '.mat'
        download_from_web(base_url + file_grp + '/files/' + _fname,
                          working_dir / 'download' / _fname, row['md5'])

        if len(row['supplemental']) > 0:
            _fname = row['filename'] + '.nwb'
            download_from_web(base_url + row['supplemental'] + '/files/' + _fname,
                              working_dir / 'download' / _fname, row['supp_md5'])
