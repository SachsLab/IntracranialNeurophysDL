import csv
from pathlib import Path
from data.utils import download_from_web


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
