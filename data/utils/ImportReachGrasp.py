# This is not currently used by any preprocessing scripts or notebooks.
# I am simply storing it here until/if I ever need it again.

import logging
import numpy as np
from neuropype.engine.packet import Chunk
from neuropype.engine.block import Block
from neuropype.engine.axes import InstanceAxis, instance
from neuropype.engine.constants import Licenses, Flags
from neuropype.engine.packet import Packet
from neuropype.engine.node import Node, Description
from neuropype.engine.ports import DataPort, StringPort, EnumPort
from neuropype.utilities.cloud import storage


logger = logging.getLogger(__name__)


class ImportReachGrasp(Node):
    # --- Input/output ports ---
    data = DataPort(Packet, "Raw data loaded with ImportNSX", required=True,
                    editable=False, mutating=True)

    filename = StringPort("", """Name of the event dataset.
                    """, is_filename=True)

    # options for cloud-hosted files
    cloud_host = EnumPort("Default", ["Default", "Azure", "S3", "Google",
                                      "Local", "None"], """Cloud storage host to
            use (if any). You can override this option to select from what kind of
            cloud storage service data should be downloaded. On some environments
            (e.g., on NeuroScale), the value Default will be map to the default
            storage provider on that environment.""")
    cloud_account = StringPort("", """Cloud account name on storage provider
            (use default if omitted). You can override this to choose a non-default
            account name for some storage provider (e.g., Azure or S3.). On some
            environments (e.g., on NeuroScale), this value will be
            default-initialized to your account.""")
    cloud_bucket = StringPort("", """Cloud bucket to read from (use default if
            omitted). This is the bucket or container on the cloud storage provider
            that the file would be read from. On some environments (e.g., on
            NeuroScale), this value will be default-initialized to a bucket
            that has been created for you.""")
    cloud_credentials = StringPort("", """Secure credential to access cloud data
            (use default if omitted). These are the security credentials (e.g.,
            password or access token) for the the cloud storage provider. On some
            environments (e.g., on NeuroScale), this value will be
            default-initialized to the right credentials for you.""")

    @classmethod
    def description(cls):
        return Description(name='Load behavior data for a Reach and Grasp dataset',
                           description="""
                           https://web.gin.g-node.org/INT/multielectrode_grasp/src/master/code/reachgraspio/reachgraspio.py
                           """,
                           version='0.1',
                           license=Licenses.MIT)

    @data.setter
    def data(self, pkt):
        if pkt is not None and 'events' in pkt.chunks:
            blk = pkt.chunks['events'].block

            # filename = storage.cloud_get(self.filename, host=self.cloud_host,
            #                              account=self.cloud_account,
            #                              bucket=self.cloud_bucket,
            #                              credentials=self.cloud_credentials)
            # logger.info("Loading behavior data from %s..." % filename)
            # import odml.tools
            # odmldoc = odml.tools.xmlparser.load(filename)

            ev_times = blk.axes[instance].times
            if len(ev_times) > 0:
                # Convert instance axis data to event marker strings
                ev_strs = blk.axes[instance].data
                for ev_ix, ev_str in enumerate(ev_strs):
                    if ev_str == 'digital_input_port':
                        ev_strs[ev_ix] = self.event_labels_str[str(blk.data[ev_ix])]

                marker_block = Block(data=np.nan * np.ones_like(ev_times),
                                     axes=(InstanceAxis(ev_times,
                                                        data=np.asarray(ev_strs),
                                                        instance_type='markers'),
                                           )
                                     )
                marker_props = {Flags.has_markers: True}
                pkt.chunks.update({'markers': Chunk(block=marker_block, props=marker_props)})
                if 'events' in pkt.chunks:
                    del pkt.chunks['events']

        self._data = pkt

    """
        Attributes:
            condition_str (dict):
                Dictionary containing a list of string codes reflecting the trial
                types that occur in recordings in a certain condition code
                (dictionary keys). For example, for condition 1 (all grip first
                conditions), condition_str[1] contains the list
                ['SGHF', 'SGLF', 'PGHF', 'PGLF'].
                Possible conditions:
                   0:[]
                     No trials, or condition not conclusive from file
                   4 types (two_cues_task):
                     1: all grip-first trial types with two different cues
                     2: all force-first trial types with two different cues
                   2 types (two_cues_task):
                     11: grip-first, but only LF types
                     12: grip-first, but only HF types
                     13: grip-first, but only SG types
                     14: grip-first, but only PG types
                   2 types (two_cues_task):
                     21: force-first, but only LF types
                     22: force-first, but only HF types
                     23: force-first, but only SG types
                     24: force-first, but only PG types
                   1 type (two_cues_task):
                     131: grip-first, but only SGLF type
                     132: grip-first, but only SGHF type
                     141: grip-first, but only PGLF type
                     142: grip-first, but only PGHF type
                     213: force-first, but only LFSG type
                     214: force-first, but only LFPG type
                     223: force-first, but only HFSG type
                     224: force-first, but only HFPG type
                   1 type (one_cue_task):
                     133: SGSG, only grip info, force unknown
                     144: PGPG, only grip info, force unknown
                     211: LFLF, only force info, grip unknown
                     222: HFHF, only force info, grip unknown
            event_labels_str (dict):
                Provides a text label for each digital event code returned as
                events by the parent BlackrockIO. For example,
                event_labels_str['65296'] contains the string 'TS-ON'.
            event_labels_codes (dict):
                Reverse of `event_labels_str`: Provides a list of event codes
                related to a specific text label for a trial event. For example,
                event_labels_codes['TS-ON'] contains the list ['65296']. In
                addition to the detailed codes, for convenience the meta codes
                'CUE/GO', 'RW-ON', and 'SR' summarizing a set of digital events are
                defined for easier access.
            trial_const_sequence_str (dict):
                Dictionary contains the ordering of selected constant trial events
                for correct trials, e.g., as TS is the first trial event in a
                correct trial, trial_const_sequence_codes['TS'] is 0.
            trial_const_sequence_codes (dict):
                Reverse of trial_const_sequence_str: Dictionary contains the
                ordering of selected constant trial events for correct trials,
                e.g., trial_const_sequence_codes[0] is 'TS'.
            performance_str (dict):
                Text strings to help interpret the performance code of a trial. For
                example, correct trials have a performance code of 255, and thus
                performance_str[255] == 'correct_trial'
            performance_codes (dict):
                Reverse of performance_const_sequence_str. Returns the performance
                code of a given text string indicating trial performance. For
                example, performance_str['correct_trial'] == 255
        """

    # Create a dictionary of conditions (i.e., the trial types presented in a
    # given recording session)
    condition_str = {
        0: [],
        1: ['SGHF', 'SGLF', 'PGHF', 'PGLF'],
        2: ['HFSG', 'HFPG', 'LFSG', 'LFPG'],
        11: ['SGLF', 'PGLF'],
        12: ['SGHF', 'PGHF'],
        13: ['SGHF', 'SGLF'],
        14: ['PGHF', 'PGLF'],
        21: ['LFSG', 'LFPG'],
        22: ['HFSG', 'HFPG'],
        23: ['HFSG', 'LFSG'],
        24: ['HFPG', 'LFPG'],
        131: ['SGLF'],
        132: ['SGHF'],
        133: ['SGSG'],
        141: ['PGLF'],
        142: ['PGHF'],
        144: ['PGPG'],
        211: ['LFLF'],
        213: ['LFSG'],
        214: ['LFPG'],
        222: ['HFHF'],
        223: ['HFSG'],
        224: ['HFPG']}

    ###########################################################################
    # event labels, the corresponding first 8 digits of their binary
    # representation and their meaning
    #
    #         R L T T L L L L
    #         w E a r E E E E
    #         P D S S D D D D                                               in
    #         u c w t b t t b                                               mo-
    #                 l r l r                                               nk-
    # label:| ^ ^ ^ ^ ^ ^ ^ ^ | status of devices:    | trial event label:| ey
    # 65280 < 0 0 0 0 0 0 0 0 > TS-OFF                > TS-OFF/STOP       > L,T
    # 65296 < 0 0 0 1 0 0 0 0 > TS-ON                 > TS-ON             > all
    # 65312 < 0 0 1 0 0 0 0 0 > TaSw                  > STOP              > all
    # 65344 < 0 1 0 0 0 0 0 0 > LEDc       (+TS-OFF)  > WS-ON/CUE-OFF     > L,T
    # 65349 < 0 1 0 0 0 1 0 1 > LEDc|rt|rb (+TS-OFF)  > PG-ON (CUE/GO-ON) > L,T
    # 65350 < 0 1 0 0 0 1 1 0 > LEDc|tl|tr (+TS-OFF)  > HF-ON (CUE/GO-ON) > L,T
    # 65353 < 0 1 0 0 1 0 0 1 > LEDc|bl|br (+TS-OFF)  > LF-ON (CUE/GO-ON) > L,T
    # 65354 < 0 1 0 0 1 0 1 0 > LEDc|lb|lt (+TS-OFF)  > SG-ON (CUE/GO-ON) > L,T
    # 65359 < 0 1 0 0 1 1 1 1 > LEDall                > ERROR-FLASH-ON    > L,T
    # 65360 < 0 1 0 1 0 0 0 0 > LEDc       (+TS-ON)   > WS-ON/CUE-OFF     > N
    # 65365 < 0 1 0 1 0 1 0 1 > LEDc|rt|rb (+TS-ON)   > PG-ON (CUE/GO-ON) > N
    # 65366 < 0 1 0 1 0 1 1 0 > LEDc|tl|tr (+TS-ON)   > HF-ON (CUE/GO-ON) > N
    # 65369 < 0 1 0 1 1 0 0 1 > LEDc|bl|br (+TS-ON)   > LF-ON (CUE/GO-ON) > N
    # 65370 < 0 1 0 1 1 0 1 0 > LEDc|lb|lt (+TS-ON)   > SG-ON (CUE/GO-ON) > N
    # 65376 < 0 1 1 0 0 0 0 0 > LEDc+TaSw             > GO-OFF/RW-OFF     > all
    # 65381 < 0 1 1 0 0 1 0 1 > TaSw (+LEDc|rt|rb)    > SR (+PG)          > all
    # 65382 < 0 1 1 0 0 1 1 0 > TaSw (+LEDc|tl|tr)    > SR (+HF)          > all
    # 65383 < 0 1 1 0 0 1 1 1 > TaSw (+LEDc|rt|rb|tl) > SR (+PGHF/HFPG)   >
    # 65385 < 0 1 1 0 1 0 0 1 > TaSw (+LEDc|bl|br)    > SR (+LF)          > all
    # 65386 < 0 1 1 0 1 0 1 0 > TaSw (+LEDc|lb|lt)    > SR (+SG)          > all
    # 65387 < 0 1 1 0 1 0 1 1 > TaSw (+LEDc|lb|lt|br) > SR (+SGLF/LGSG)   >
    # 65389 < 0 1 1 0 1 1 0 1 > TaSw (+LEDc|rt|rb|bl) > SR (+PGLF/LFPG)   >
    # 65390 < 0 1 1 0 1 1 1 0 > TaSw (+LEDc|lb|lt|tr) > SR (+SGHF/HFSG)   >
    # 65391 < 0 1 1 0 1 1 1 1 > LEDall (+TaSw)        > ERROR-FLASH-ON    > L,T
    # 65440 < 1 0 1 0 0 0 0 0 > RwPu (+TaSw)          > RW-ON (noLEDs)    > N
    # 65504 < 1 1 1 0 0 0 0 0 > RwPu (+LEDc)          > RW-ON (-CONF)     > L,T
    # 65509 < 1 1 1 0 0 1 0 1 > RwPu (+LEDcr)         > RW-ON (+CONF-PG)  > all
    # 65510 < 1 1 1 0 0 1 1 0 > RwPu (+LEDct)         > RW-ON (+CONF-HF)  > N?
    # 65513 < 1 1 1 0 1 0 0 1 > RwPu (+LEDcb)         > RW-ON (+CONF-LF)  > N?
    # 65514 < 1 1 1 0 1 0 1 0 > RwPu (+LEDcl)         > RW-ON (+CONF-SG)  > all
    #         ^ ^ ^ ^ ^ ^ ^ ^
    #        label binary code
    #
    # ABBREVIATIONS:
    # c (central), l (left), t (top), b (bottom),  r (right),
    # HF (high force, LEDt), LF (low force, LEDb), SG (side grip, LEDl),
    # PG (precision grip, LEDr), RwPu (reward pump), TaSw (table switch),
    # TS (trial start), SR (switch release), WS (warning signal), RW (reward),
    # L (Lilou), T (Tanya t+a), N (Nikos n+i)
    ###########################################################################

    # Create dictionaries for event labels
    event_labels_str = {
        '65280': 'TS-OFF/STOP',
        '65296': 'TS-ON',
        '65312': 'STOP',
        '65344': 'WS-ON/CUE-OFF',
        '65349': 'PG-ON',
        '65350': 'HF-ON',
        '65353': 'LF-ON',
        '65354': 'SG-ON',
        '65359': 'ERROR-FLASH-ON',
        '65360': 'WS-ON/CUE-OFF',
        '65365': 'PG-ON',
        '65366': 'HF-ON',
        '65369': 'LF-ON',
        '65370': 'SG-ON',
        '65376': 'GO/RW-OFF',
        '65381': 'SR (+PG)',
        '65382': 'SR (+HF)',
        '65383': 'SR (+PGHF/HFPG)',
        '65385': 'SR (+LF)',
        '65386': 'SR (+SG)',
        '65387': 'SR (+SGLF/LFSG)',
        '65389': 'SR (+PGLF/LFPG)',
        '65390': 'SR (+SGHF/HFSG)',
        '65391': 'ERROR-FLASH-ON',
        '65440': 'RW-ON (noLEDs)',
        '65504': 'RW-ON (-CONF)',
        '65509': 'RW-ON (+CONF-PG)',
        '65510': 'RW-ON (+CONF-HF)',
        '65513': 'RW-ON (+CONF-LF)',
        '65514': 'RW-ON (+CONF-SG)'}
    event_labels_codes = dict(
        [(k, []) for k in np.unique(list(event_labels_str.values()))])
    for k in event_labels_codes.keys():
        for l, v in event_labels_str.items():
            if v == k:
                event_labels_codes[k].append(l)

    # additional summaries
    event_labels_codes['CUE/GO'] = \
        event_labels_codes['SG-ON'] + \
        event_labels_codes['PG-ON'] + \
        event_labels_codes['LF-ON'] + \
        event_labels_codes['HF-ON']
    event_labels_codes['RW-ON'] = \
        event_labels_codes['RW-ON (+CONF-PG)'] + \
        event_labels_codes['RW-ON (+CONF-HF)'] + \
        event_labels_codes['RW-ON (+CONF-LF)'] + \
        event_labels_codes['RW-ON (+CONF-SG)'] + \
        event_labels_codes['RW-ON (-CONF)'] + \
        event_labels_codes['RW-ON (noLEDs)']
    event_labels_codes['SR'] = \
        event_labels_codes['SR (+PG)'] + \
        event_labels_codes['SR (+HF)'] + \
        event_labels_codes['SR (+LF)'] + \
        event_labels_codes['SR (+SG)'] + \
        event_labels_codes['SR (+PGHF/HFPG)'] + \
        event_labels_codes['SR (+SGHF/HFSG)'] + \
        event_labels_codes['SR (+PGLF/LFPG)'] + \
        event_labels_codes['SR (+SGLF/LFSG)']
    del k, l, v

    # Create dictionaries for constant trial sequences (in all monkeys)
    # (bit position (value) set if trial event (key) occurred)
    trial_const_sequence_codes = {
        'TS-ON': 0,
        'WS-ON': 1,
        'CUE-ON': 2,
        'CUE-OFF': 3,
        'GO-ON': 4,
        'SR': 5,
        'RW-ON': 6,
        'STOP': 7}
    trial_const_sequence_str = dict(
        (v, k) for k, v in trial_const_sequence_codes.items())

    # Create dictionaries for trial performances
    # (resulting decimal number from binary number created from trial_sequence)
    performance_codes = {
        'incomplete_trial': 0,
        'error<SR-ON': 159,
        'error<WS': 161,
        'error<CUE-ON': 163,
        'error<CUE-OFF': 167,
        'error<GO-ON': 175,
        'grip_error': 191,
        'correct_trial': 255}
    performance_str = dict((v, k) for k, v in performance_codes.items())

