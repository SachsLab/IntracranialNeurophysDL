from neuropype.engine import *


class ImportKJMFingerFlex(Node):
    """Load fingerflex data file from KJM dataset.
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
        import pandas as pd
        from ..utils import AREA_LABELS

        srate = 1000.

        dat_contents = scipy.io.loadmat(self.filename, squeeze_me=True)
        # brain = dat_contents['brain']  # rec array with fields vert, tri

        sub_path = Path(self.filename).parent
        participant_id = sub_path.name
        sub_stim = sub_path / (participant_id + '_stim.mat')
        stim_contents = scipy.io.loadmat(sub_stim, squeeze_me=True)

        # Time vector
        tvec = np.arange(dat_contents['data'].shape[0]) / srate

        # Process the cue and stimulus to get an events chunk
        stim_map = {-2: 'Unknown', -1: 'Transition', 0: 'ISI',
                    1: 'Thumb', 2: 'Index', 3: 'Middle', 4: 'Ring', 5: 'Little'}
        ev_dict = {'Marker': [], 'MarkerType': [], 'Time': []}
        ev_t = []
        # Cues
        cue = dat_contents['cue']  # samples x _, int
        b_cue_onset = np.diff(np.hstack((0, cue))) != 0
        ev_dict['Marker'].extend([stim_map[_] for _ in cue[b_cue_onset]])
        ev_dict['MarkerType'].extend(['Cue'] * np.sum(b_cue_onset))
        ev_dict['Time'].extend(tvec[b_cue_onset].tolist())
        # Stims
        stim = stim_contents['stim']  # samples x _, int16
        b_stim_onset = np.diff(np.hstack((0, stim))) != 0
        ev_dict['Marker'].extend([stim_map[_] for _ in stim[b_stim_onset]])
        ev_dict['MarkerType'].extend(['Stim'] * np.sum(b_stim_onset))
        ev_dict['Time'].extend(tvec[b_stim_onset].tolist())
        ev_df = pd.DataFrame(ev_dict).sort_values(by=['Time'])
        if False:
            import matplotlib.pyplot as plt
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(tvec, cue)
            for t_ev in tvec[b_cue_onset]:
                ax1.axvline(t_ev, color='k', linestyle='--')
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.plot(tvec, stim)
            for t_ev in tvec[b_stim_onset]:
                ax2.axvline(t_ev, color='k', linestyle='--')
        # Combine into instance axis data table and create chunk
        ev_ax = InstanceAxis(ev_df['Time'].values, data=ev_df.drop(columns=['Time']).to_records(index=False))
        ev_chunk = Chunk(block=Block(data=np.nan*np.ones(len(ev_ax),), axes=(ev_ax,)),
                         props=[Flags.is_event_stream])

        # Process signals
        # Get the channel labels and locations.
        locs = dat_contents['locs']  # channels x (x,y,z)
        region_map = {0: 'Unknown', 1: 'Dorsal_M1', 2: 'UndefinedA', 3: 'Dorsal_S1', 4: 'Ventral_Sensorimotor',
                      5: 'UndefinedB',
                      6: 'Frontal', 7: 'Parietal', 8: 'Temporal', 9: 'Occipital'}
        elec_regions = dat_contents['elec_regions']  # channels x _, int

        # Append a .N to each electrode name, where N is the count of electrodes with that name.
        # The below method is a little silly, but more straightforward approaches did not work
        # in interactive debug mode.
        name_counts = {_: 0 for _ in region_map.keys()}
        elec_names = []
        for elec_ix, reg_ix in enumerate(elec_regions):
            elec_names.append('{}.{}'.format(region_map[reg_ix], name_counts[reg_ix]))
            name_counts[reg_ix] += 1

        # Put the data into a data chunk
        # samples x channels; int; 1 unit = 0.0298 uV; 1-pole bandpassed 0.15-200
        data_chunk = Chunk(block=Block(data=0.0298 * dat_contents['data'],
                                       axes=(TimeAxis(times=tvec, nominal_rate=srate),
                                             SpaceAxis(names=elec_names, positions=locs))),
                           props=[Flags.is_signal])
        data_chunk.props['source_url'] = 'file://' + str(self.filename)

        # Finger flexion data.
        flex_chunk = Chunk(block=Block(data=dat_contents['flex'],
                                       axes=(TimeAxis(times=tvec, nominal_rate=srate),
                                             SpaceAxis(names=['Thumb', 'Index', 'Middle', 'Ring', 'Little']))),
                           props=[Flags.is_signal])
        flex_chunk.props['source_url'] = 'file://' + str(self.filename)

        self._data = Packet(chunks=OrderedDict({'markers': ev_chunk, 'signals': data_chunk, 'behav': flex_chunk}))

    def on_port_assigned(self):
        """Callback to reset internal state when a value was assigned to a
        port (unless the port's setter has been overridden)."""
        self._has_emitted = False
        self.signal_changed(True)

    def is_finished(self):
        """Whether this node is finished producing output."""
        return not self.filename or self._has_emitted
