import csv
from pathlib import Path
# Some hacks to fix the directory and PATH so this can work from IDE or from console.
if Path.cwd().stem == 'joeyo':
    import os
    os.chdir('../..')
import sys
sys.path.insert(0, ".")
import data.utils as indl_du


if __name__ == "__main__":
    ROW_RANGE = [13, 19]  # Use [0, np.inf] to process all rows.

    data_dir = Path.cwd() / 'data' / 'joeyo'
    datasets_file = data_dir / 'datasets.csv'
    # Get the list of datasets to download
    datasets = []
    with open(datasets_file) as csvfile:
        datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in datasetreader:
            datasets.append(row)

    # Create a local folder to store the data
    local_dir = data_dir / 'download'
    if not local_dir.is_dir():
        local_dir.mkdir()

    base_url = 'https://zenodo.org/record/'
    file_grp = '583331'
    for row_ix, row in enumerate(datasets):
        if row_ix < ROW_RANGE[0] or row_ix > ROW_RANGE[1]:
            continue
        _fname = row['filename'] + '.mat'
        indl_du.download_from_web(base_url + file_grp + '/files/' + _fname,
                                  data_dir / 'download' / _fname, row['md5'])

        if len(row['supplemental']) > 0:
            _fname = row['filename'] + '.nwb'
            indl_du.download_from_web(base_url + row['supplemental'] + '/files/' + _fname,
                                      data_dir / 'download' / _fname, row['supp_md5'])
