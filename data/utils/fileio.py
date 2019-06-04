import numpy as np
from pathlib import Path


def _recurse_get_dict_from_group(grp):
    result = dict(grp.attrs)
    for k, v in result.items():
        if isinstance(v, np.ndarray) and v.dtype.char == 'S':
            result[k] = v.astype('U').astype('O')
    for k, v in grp.items():
        result[k] = _recurse_get_dict_from_group(v)
    return result


def from_neuropype_h5(filename):
    import h5py
    from pandas import DataFrame
    f = h5py.File(filename, 'r')

    chunks = []
    if 'chunks' in f.keys():
        chunks_group = f['chunks']
        for ch_key in chunks_group.keys():
            chunk_group = chunks_group.get(ch_key)
            # Process data
            block_group = chunk_group.get('block')
            data_ = block_group.get('data')
            if isinstance(data_, h5py.Dataset):
                data = data_[()]
            else:
                # Data is a group. This only happens with sparse matrices.
                import scipy.sparse
                data = scipy.sparse.csr_matrix((data_['data'][:], data_['indices'][:], data_['indptr'][:]),
                                               data_.attrs['shape'])

            axes_group = block_group.get('axes')
            axes = []
            for ax_ix, axis_key in enumerate(axes_group.keys()):
                axis_group = axes_group.get(axis_key)
                ax_type = axis_group.attrs.get('type')
                new_ax = {'name': axis_key, 'type': ax_type}
                if ax_type == 'axis':
                    new_ax.update(dict(x=np.arange(data.shape[ax_ix])))
                elif ax_type == 'time':
                    nom_rate = axis_group.attrs.get('nominal_rate')
                    if np.isnan(nom_rate):
                        nom_rate = None
                    new_ax.update(dict(nominal_rate=nom_rate,
                                       times=axis_group.get('times')[()]))
                elif ax_type == 'frequency':
                    new_ax.update(dict(frequencies=axis_group.get('frequencies')[()]))
                elif ax_type == 'space':
                    new_ax.update(dict(names=axis_group.get('names')[()],
                                       naming_system=axis_group.attrs['naming_system'],
                                       positions=axis_group.get('positions')[()],
                                       coordinate_system=axis_group.attrs['coordinate_system'],
                                       units=axis_group.get('units')[()]))
                elif ax_type == 'feature':
                    new_ax.update(dict(names=axis_group.get('names')[()],
                                       units=axis_group.get('units')[()],
                                       properties=axis_group.get('properties')[()],
                                       error_distrib=axis_group.get('error_distrib')[()],
                                       sampling_distrib=axis_group.get('sampling_distrib')[()]))
                elif ax_type == 'instance':
                    new_ax.update({'times': axis_group.get('times')[()]})
                    if 'instance_type' in axis_group.attrs:
                        new_ax.update({'instance_type': axis_group.attrs['instance_type']})
                    _dat = axis_group.get('data')[()]
                    if not _dat.dtype.names:
                        new_ax.update({'data': axis_group.get('data')[()]})
                    else:
                        _df = DataFrame(_dat)
                        # Convert binary objects to string objects
                        str_df = _df.select_dtypes([np.object])
                        str_df = str_df.stack().str.decode('utf-8').unstack()
                        for col in str_df:
                            _df[col] = str_df[col]
                        new_ax.update({'data': _df})

                elif ax_type == 'statistic':
                    new_ax.update(dict(param_types=axis_group.get('param_types')[()]))
                elif ax_type == 'lag':
                    new_ax.update(dict(xlags=axis_group.get('lags')[()]))
                if new_ax is not None:
                    axes.append(new_ax)

            chunks.append((ch_key, dict(data=data, axes=axes,
                                        props=_recurse_get_dict_from_group(chunk_group.get('props')))))

    return chunks


def load_joeyo_reaching(data_path, sess_id, x_chunk='lfps', zscore=True):
    """
    Load data from the joeyo dataset.
    :param data_path: path to joeyo data dir (i.e., parent of 'converted'
    :param sess_id: 'indy_2016' + one of '0921_01', '0927_04', '0927_06', '0930_02', '0930_05' '1005_06' '1006_02'
    :param x_chunk: 'lfps', 'spikerates', or 'spiketimes'
    :return: X, Y, X_ax_info, Y_ax_info
    """
    file_path = Path(data_path) / 'converted' / (sess_id + '.h5')
    chunks = from_neuropype_h5(file_path)
    chunk_names = [_[0] for _ in chunks]

    Y_chunk = chunks[chunk_names.index('behav')][1]
    Y_ax_types = [_['type'] for _ in Y_chunk['axes']]
    Y_ax_info = {'channel_names': Y_chunk['axes'][Y_ax_types.index('space')]['names'],
                 'timestamps': Y_chunk['axes'][Y_ax_types.index('time')]['times'],
                 'fs': Y_chunk['axes'][Y_ax_types.index('time')]['nominal_rate']}

    X_chunk = chunks[chunk_names.index(x_chunk)][1]
    X_ax_types = [_['type'] for _ in X_chunk['axes']]
    X_ax_info = {'channel_names': X_chunk['axes'][X_ax_types.index('space')]['names'],
                 'timestamps': X_chunk['axes'][X_ax_types.index('time')]['times'],
                 'fs': X_chunk['axes'][X_ax_types.index('time')]['nominal_rate']}

    if zscore:
        X_chunk['data'] = (X_chunk['data'] - np.mean(X_chunk['data'], axis=1, keepdims=True))\
                          / np.std(X_chunk['data'], axis=1, keepdims=True)
        Y_chunk['data'] = (Y_chunk['data'] - np.mean(Y_chunk['data'], axis=1, keepdims=True)) \
                          / np.std(Y_chunk['data'], axis=1, keepdims=True)

    return X_chunk['data'], Y_chunk['data'], X_ax_info, Y_ax_info


def load_faces_houses(data_path, sub_id, feature_set='full'):
    """
    Load a file from the KJM faces_basic dataset.
    :param data_path: Path object pointing to the root of the data directory (parent of 'converted/faces_basic')
    :param sub_id: Subject ID (2-character string)
    :param feature_set: 'full' or 'bp' for band-power
    :return: X, Y, ax_info
    """
    test_file = data_path / 'converted' / 'faces_basic' / (sub_id + '_' + feature_set + '.h5')
    chunks = from_neuropype_h5(test_file)
    chunk_names = [_[0] for _ in chunks]
    chunk = chunks[chunk_names.index('signals')][1]
    ax_types = [_['type'] for _ in chunk['axes']]
    instance_axis = chunk['axes'][ax_types.index('instance')]
    X = chunk['data']
    Y = instance_axis['data']['Marker'].values.reshape(-1, 1)
    ax_info = {'instance_data': instance_axis['data'],
               'fs': chunk['axes'][ax_types.index('time')]['nominal_rate'],
               'timestamps': chunk['axes'][ax_types.index('time')]['times'],
               'channel_names': chunk['axes'][ax_types.index('space')]['names'],
               'channel_locs': chunk['axes'][ax_types.index('space')]['positions']
               }
    return X, Y, ax_info


if __name__ == '__main__':
    from pathlib import Path
    if Path.cwd().stem == 'utils':
        import os
        os.chdir('../..')
    # datadir = Path.cwd() / 'data' / 'kjm_ecog'
    # res_tuple = load_faces_houses(datadir, 'mv')
    datadir = Path.cwd() / 'data' / 'joeyo'
    res_tuple = load_joeyo_reaching(datadir, 'indy_20160921_01')
    print(res_tuple[2])  # X_ax_info
