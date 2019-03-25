import math
import hashlib
import requests
import tqdm


def get_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_from_web(remote_url, local_path, md5=None):
    if local_path.exists():
        if md5 is not None:
            if get_md5(local_path) == md5:
                print("{} already exists and md5 checks out. Skipping.".format(local_path))
                return
            else:
                print("{} already exists but md5 is bad. Replacing.".format(local_path))
                local_path.unlink()
        else:
            print("{} already exists. Skipping.".format(local_path))
            return
    r = requests.get(remote_url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    chunk_size = 1024
    wrote = 0
    print("Downloading to {}".format(local_path))
    with open(local_path, 'wb') as f:
        for data in tqdm.tqdm(r.iter_content(chunk_size=chunk_size), total=math.ceil(total_size // chunk_size),
                              unit='kB', unit_scale=False):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR downloading to {}. Deleting.".format(local_path))
        local_path.unlink()
    elif (md5 is not None) and (get_md5(local_path) != md5):
        print("ERROR md5 mismatch for {}. Deleting.".format(local_path))
        local_path.unlink()
    print("Finished downloading {}".format(local_path))
