#  Copyright (C) 2014-2019 Syntrogi Inc dba Intheon. All rights reserved.

# NeuroPype Sample Code (Tutorial)
#
# This example code shows how to implement a wrapper for a scikit-learn
# compatible machine learning algorithm as a NeuroPype node.

import logging
import numpy as np
from neuropype.engine import *
from neuropype.utilities.helpers import scoring_options
from neuropype.nodes.machine_learning._shared import apply_predictor


logger = logging.getLogger(__name__)


class VaryingLDA(Node):
    """Use Varying Linear Discriminant Analysis to classify data instances."""

    # --- Input/output ports ---
    data = DataPort(Packet, "Data to process.")

    # --- Properties ---
    # generally we are exposing every parameter of the algorithm here, using
    # the appropriate port type, and using the same defaults as in sklearn if
    # possible. Many of these parameters may be declared as expert-mode to keep
    # the GUI clutter-free and non-confusing for beginners.
    probabilistic = BoolPort(True, """Use probabilistic outputs. If enabled, the
        node will output for each class the probability that a given trial is of
        that class; otherwise it will output the most likely class label.
        """, verbose_name='output probabilities')
    solver = EnumPort('eigen', ['eigen', 'lsqr', 'svd'], """Solver to use.
        This node supports formulations based on a least-squares solution
        (lsqr), eigenvalue decomposition (eigen), and singular-value
        decomposition (svd). Some of these methods are known to have numerical
        issues under various circumstances -- consider trying different settings
        if you cannot achieve the desired results. Note: the svd method can
        handle many features, but does not support shrinkage-type
        regularization.""", verbose_name='solver to use', expert=True)
    class_weights = Port(None, object, """Per-class weights. Optionally this is
        a mapping from class label to weight. This is formatted as in, e.g.,
        {0: 0.1, 1: 0.2, 2: 0.1}. Class weights are used fairly infrequently,
        except when the cost of false detections of different classes needs to
        be traded off.
        """, verbose_name='per-class weight', expert=True)
    tolerance = FloatPort(0.0001, None, """Threshold for rank estimation in
        SVD. Using a larger value will more aggressively prune features, but
        can make the difference between it working or not.
        """, verbose_name='rank estimation threshold (if svd)', expert=True)
    # this parameter is customarily added to NeuroPype nodes to allow the user
    # to choose whether the model should be re-calibrated when another dataset
    # is passed through, or whether it should only perform predictions on the
    # new data
    initialize_once = BoolPort(True, """Calibrate the model only once. If set to
        False, then this node will recalibrate itself whenever a non-streaming
        data chunk is received that has both training labels and associated
        training instances.
        """, verbose_name='calibrate only once', expert=True)
    # this parameter allows you to override the NeuroPype default behavior where
    # a node's state would be cleared when something in the preceding (upstream)
    # graph changes (e.g., the scaling of features); by setting this to true,
    # you allow the user to retain the trainable state, at their own risk that
    # the trained model may not be applicable to the new data format; this is for
    # the case where the user has meticulously trained a model, but needs to make
    # a minute change in the upstream settings without losing their model
    dont_reset_model = BoolPort(False, """Do not reset the model when the
        preceding graph is changed. Normally, when certain parameters of
        preceding nodes are being changed, the model will be reset. If this is
        enabled, the model will persist, but there is a chance that the model
        is incompatible when input data format to this node has changed.
        """, verbose_name='do not reset model', expert=True)
    independent_axis = EnumPort(default="time", domain=list(axis_names).remove('instance'), help="""
        Axis to isolate. LDA will be performed for each entry in this axis independently.
        For example, if this is set to 'time', then LDA will be performed for each time point independently.
        The final LDA scores come from summing the LDA across entries in this axis for each class.
        """, verbose_name='LDA performed independently for each entry in this axis.')
    verbosity = IntPort(0, None, """Verbosity level. Higher numbers will produce
        more extensive diagnostic output.""", verbose_name='verbosity level')
    cond_field = StringPort(default='TargetValue', help="""The name of the
        instance data field that contains the conditions to be discriminated.
        This parameter will be ignored if the packet has previously been processed
        by a BakeDesignMatrix node.""", expert=True)
    n_components = IntPort(default=None, help="""The number of components to keep in the
        reduced model""")

    def __init__(self, probabilistic: Union[bool, None, Type[Keep]] = Keep, solver: Union[str, None, Type[Keep]] = Keep, class_weights: Union[object, None, Type[Keep]] = Keep, tolerance: Union[float, None, Type[Keep]] = Keep, initialize_once: Union[bool, None, Type[Keep]] = Keep, dont_reset_model: Union[bool, None, Type[Keep]] = Keep, verbosity: Union[int, None, Type[Keep]] = Keep, cond_field: Union[str, None, Type[Keep]] = Keep, n_components: Union[int, None, Type[Keep]] = Keep, **kwargs):
        """Create a new node. Accepts initial values for the ports."""
        # unlike many other NeuroPype nodes, machine learning nodes usually do
        # not support multiple parallel data streams (these are supposed to have
        # been merged into a joint feature vector by this point); the only
        # exception is that there may be a data stream containing the labels if
        # the model shall be retrained; still, the state holds only a single model
        self.M = {}  # predictive model
        super().__init__(probabilistic=probabilistic, solver=solver, class_weights=class_weights, tolerance=tolerance, initialize_once=initialize_once, dont_reset_model=dont_reset_model, verbosity=verbosity, cond_field=cond_field, n_components=n_components, **kwargs)

    @classmethod
    def description(cls):
        """Declare descriptive information about the node."""
        return Description(name='Varying Linear Discriminant Analysis',
                           description="""\
                           """,
                           version='0.1.0', status=DevStatus.alpha)

    @data.setter
    def data(self, v):
        # this call is the canonical way to get the training data and optionally
        # training labels from the given Packet v; if one or both of these items
        # are missing, the respective variable will be None
        X, y, X_n = extract_chunks(v, collapse_features=False, y_column=self.cond_field, return_data_chunk_label=True)
        # determine whether the model shall be trained
        init_flag = (not self.initialize_once) or (X_n not in self.M)
        # check if all conditions are met to (re)train
        if X is not None and y is not None and init_flag:
            # generally we're deferring heavy imports until they're actually
            # needed to reduce the memory footprint of NeuroPype, as well as
            # the startup time
            from sklearn.discriminant_analysis import \
                LinearDiscriminantAnalysis as LDA

            # (we only need this because of an inconsistency in how prior
            # class weights in LDA happen to be handled)
            if self.class_weights is not None:
                # noinspection PyUnresolvedReferences
                # LDA requires an array here instead of a dict
                priors = [self.class_weights[i]
                          if i in self.class_weights else 1.0
                          for i in range(max(self.class_weights.keys()) + 1)]
            else:
                priors = None

            if X.block.ndim > 3:
                view = X.block[axis_definers[self.independent_axis], instance, collapsedaxis]
            else:
                # Keep the last axis in its native type.
                view = X.block[axis_definers[self.independent_axis], instance, ...]
            n_models, n_trials, n_features = view.shape

            # Train an independent model for each entry in the self.independent_axis
            logger.info("Now training {} LDAs, 1 for each element in {}.".format(n_models, view.axes[0].type_str))
            models = []
            for m_ix in range(n_models):
                # Initialize the model
                temp = LDA(solver=self.solver, priors=priors, tol=self.tolerance)
                # finally fit the model given the data -- this line assumes that
                # the predicted value is one-dimensional (per trial, so it's a vector over trials)
                temp.fit(view.data[m_ix], y.reshape(-1))
                # Save the result
                models.append(temp)

            # TODO: First output axis should be classes (i.e., conditional mean of instance axis.)
            out_axes = (InstanceAxis(np.arange(len(models[0].classes_)), models[0].classes_),
                        view.axes[0]) + view.axes[2:]
            self.M[X_n] = {
                'models': models,
                'filters': np.eye(n_features),
                'patterns': np.eye(n_features),
                'axes': out_axes
            }

            n_comps = min(self.n_components or np.Inf, n_models)
            if n_comps < n_models:
                # We can decompose the model weights to get a dimensionality-reduced model
                import scipy.linalg
                from sklearn.utils.extmath import svd_flip
                # For the decomposed (dim-reduced) model, we will save the ...
                ind_weights = []  # independent axis weights (e.g., weights over time in time-varying LDA)
                filters = []      # the 'other' axis weights (if space, this is a spatial filter)
                patterns = []     # the patterns are the same as filters in PCA because orthogonality gives X = inv(X).T
                coefs = []        # the matrix product of ind_weights and filters
                weights = np.stack([_.coef_ for _ in models])
                for class_ix in range(weights.shape[1]):
                    Wy = weights[:, class_ix, :]
                    # A PCA would center Wy first, otherwise this is identical.
                    u, s, vh = scipy.linalg.svd(Wy, full_matrices=False)
                    u, vh = svd_flip(u, vh)  # flip eigenvectors' sign to enforce deterministic output
                    # assert np.allclose(Wy, np.dot(u[:, :n_models] * s[:n_models], vh[:n_models, :]))
                    ind_weights.append(u[:, :n_comps] * s[:n_comps])
                    filters.append(vh[:n_comps, :])  # same as pca.components_
                    coefs.append(np.dot(ind_weights[-1], filters[-1]))
                    # PCA filters are always orthogonal, so that means filter.T is equal to its inverse...
                    # which means that the patterns are equal to the filters.
                    # I'll leave the calculation below as an example for what should be done when not orthogonal.
                    vh_inv = np.linalg.pinv(vh)
                    patterns.append(np.transpose(vh_inv)[:n_comps, :])

                # Put the coefficients back into LDA models
                coefs = np.stack(coefs)
                for m_ix in range(n_models):
                    models[m_ix].coef_ = coefs[:, m_ix, :]

                # Save the result
                self.M[X_n].update({
                    'models': models,
                    'ind_weights': np.stack(ind_weights),
                    'filters': np.stack(filters),
                    'patterns': np.stack(patterns)
                })

        #
        X_view = X.block[axis_definers[self.independent_axis], instance, collapsedaxis]
        n_models, n_trials, n_features = X_view.shape
        models = self.M[X_n]['models']
        n_classes = models[0].coef_.shape[0]

        names = models[0].classes_.tolist()
        proba_ax = FeatureAxis(names=names,
                               properties=[ValueProperty.NORMALIZED] * n_classes,
                               sampling_distrib=[DistributionType.BERNOULLI] * n_classes)
        out_block = Block(data=np.zeros((n_trials, n_classes)),
                          axes=(X_view.axes[1], proba_ax))
        for m_ix, X in enumerate(X_view.data):
            out_block.data += np.dot(X, models[m_ix].coef_.T) + models[m_ix].intercept_
        out_block.data /= n_models

        # Convert to probabilities
        out_block.data *= -1
        np.exp(out_block.data, out_block.data)
        out_block.data += 1
        np.reciprocal(out_block.data, out_block.data)
        if n_classes == 2:  # binary case
            out_block.data = np.column_stack([1 - out_block.data, out_block.data])
        else:
            # OvR normalization, like LibLinear's predict_probability
            out_block.data /= out_block.data.sum(axis=1).reshape((out_block.data.shape[0], -1))

        v.chunks[X_n].block = out_block

        # finally we write the updated packet into our ._data variable, which is
        # the one that will be read out when our .data property is read from
        # (i.e., output data is being transferred out of this node)
        self._data = v

    def on_signal_changed(self):
        """Callback to reset internal state when an input wire has been
        changed."""
        if not self.dont_reset_model:
            self.M = None

    def on_port_assigned(self):
        """Callback to reset internal state when a value was assigned to a
        port (unless the port's setter has been overridden)."""
        self.signal_changed(True)

    def get_model(self):
        """Get the trainable model parameters of the node."""
        # any node that has "trainable" state (i.e., that should be possible to
        # save and load) should expose the get_model and set_model methods as
        # done here
        return {'M': self.M}

    def set_model(self, v):
        """Set the trainable model parameters of the node."""
        self.M = v['M']
