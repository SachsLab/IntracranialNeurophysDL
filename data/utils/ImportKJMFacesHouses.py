from neuropype.engine import *


class ImportKJMFacesHouses(Node):
    """Load faces houses data file from KJM dataset.
    https://exhibits.stanford.edu/data/catalog/zk881ps0522
    """

    # --- Input/output ports ---
    data = DataPort(Packet, "Output signal.", OUT)

    # --- Properties ---
    filename = StringPort("", """Path to the the file.
        """, is_filename=False)

    def __init__(self, filename: Union[str, None, Type[Keep]] = Keep, **kwargs):
        """Create a new node. Accepts initial values for the ports."""
        self._has_emitted = False
        super().__init__(filename=filename, **kwargs)

    @classmethod
    def description(cls):
        """Declare descriptive information about the node."""
        return Description(name='Import file from KJM ECoG dataset',
                           description="""\
                           Import file from Kail Miller's online ECoG dataset.""",
                           url='https://exhibits.stanford.edu/data/catalog/zk881ps0522',
                           version='1.0.0', status=DevStatus.production)

    @Node.update.setter
    def update(self, v):
        if not self.filename:
            self._data = None
            return

        if self._has_emitted:
            self._data = None
            return
        self._has_emitted = True

        from pathlib import Path
        import scipy.io
        import numpy as np
        from collections import OrderedDict
        from ..utils import AREA_LABELS

        dat_contents = scipy.io.loadmat(self.filename, squeeze_me=True)

        stim = dat_contents['stim']  # samples x _; uint8
        data = dat_contents['data']  # samples x channels; float
        srate = dat_contents['srate']

        # Time vector
        tvec = np.arange(len(stim)) / srate

        # Process the stimulus to get an events chunk
        b_stim_onset = np.diff(np.hstack((0, stim))) != 0
        b_stim_onset = np.logical_and(b_stim_onset, stim != 0)
        stim_inds = np.where(b_stim_onset)[0]
        stim_vals = stim[stim_inds]
        stim_content = np.repeat(['ISI'], len(stim_vals)).astype(object)
        stim_content[stim_vals <= 50] = 'house'
        stim_content[np.logical_and(stim_vals > 50, stim_vals <= 100)] = 'face'
        stim_ax = InstanceAxis(tvec[b_stim_onset], data=stim_content.tolist())
        stim_ax.append_fields(['StimID'], [stim_vals])
        stim_chunk = Chunk(block=Block(data=np.nan * np.ones(stim_ax.data.shape), axes=(stim_ax,)),
                           props=[Flags.is_event_stream])

        # Get the channel labels and locations.
        fn_path = Path(self.filename)
        participant_id = str(fn_path.stem).split('_')[0]
        locs_path = fn_path.parents[2] / 'locs' / (participant_id + '_xslocs.mat')
        locs_contents = scipy.io.loadmat(locs_path, squeeze_me=True)  # 'elcode' and 'locs'
        elec_names = np.array([AREA_LABELS[el_code - 1] for el_code in locs_contents['elcode']],
                              dtype=object)
        # Append a .N to each electrode name, where N is the count of electrodes with that name.
        # The below method is a little silly, but more straightforward approaches did not work
        # in interactive debug mode.
        name_counts = {_: 0 for _ in np.unique(elec_names)}
        for elec_ix, elec_name in enumerate(elec_names):
            elec_names[elec_ix] = '{}.{}'.format(elec_name.title(), name_counts[elec_name])
            name_counts[elec_name] += 1

        # Put the data into a data chunk
        data_chunk = Chunk(block=Block(data=data, axes=(TimeAxis(times=tvec, nominal_rate=srate),
                                                        SpaceAxis(names=elec_names, positions=locs_contents['locs']))),
                           props=[Flags.is_signal])
        data_chunk.props['source_url'] = 'file://' + str(self.filename)

        self._data = Packet(chunks=OrderedDict({'markers': stim_chunk, 'signals': data_chunk}))

    def on_port_assigned(self):
        """Callback to reset internal state when a value was assigned to a
        port (unless the port's setter has been overridden)."""
        self._has_emitted = False
        self.signal_changed(True)

    def is_finished(self):
        """Whether this node is finished producing output."""
        return not self.filename or self._has_emitted
