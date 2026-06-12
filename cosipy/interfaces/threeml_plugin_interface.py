from typing import Dict, Optional

from cosipy.interfaces import ThreeMLModelFoldingInterface, BackgroundInterface
from cosipy.interfaces.likelihood_interface import LikelihoodInterface
from threeML import PluginPrototype, Parameter

__all__ = ["ThreeMLPluginInterface"]

class ThreeMLPluginInterface(PluginPrototype):

    def __init__(self,
                 name: str,
                 likelihood: LikelihoodInterface,
                 response:ThreeMLModelFoldingInterface,
                 bkg:Optional[BackgroundInterface] = None,):
        """
        Parameters
        ----------
        name: str
        likefun: str or LikelihoodInterface (Use at your own
        risk. make sure it knows about the input data, response and
        bkg)

        """

        # PluginPrototype.__init__ does the following:
        # Sets _name = name
        # Sets _tag = None
        # Set self._nuisance_parameters, which we do not use because
        # we're overriding nuisance_parameters() and
        # update_nuisance_parameters()
        super().__init__(name, {})

        self._like = likelihood
        self._response = response
        self._bkg = bkg

        # Currently, the only nuisance parameters are the ones for the
        # bkg. We could have systematics here as well
        if self._bkg is None:
            self._threeml_bkg_parameters = {}
        else:
            # 1. Adds plugin name, required by 3ML code
            # See https://github.com/threeML/threeML/blob/7a16580d9d5ed57166e3b1eec3d4fccd3eeef1eb/threeML/classicMLE/joint_likelihood.py#L131
            # 2. Translation to bkg bare parameters. 3ML "Parameter"
            # keeps track of a few more things than a "bare"
            # (Quantity) parameter.
            self._threeml_bkg_parameters = {self._add_prefix_name(label): Parameter(label, param.value, unit=param.unit) for label, param in self._bkg.parameters.items()}

        # Allows idiom plugin.bkg_parameters["bkg_param_name"] to get
        # 3ML parameter
        self.bkg_parameter = ThreeMLPluginInterface._Bkg_parameter(self)

    def _add_prefix_name(self, label):
        return self._name + "_" + label

    def _remove_prefix_name(self, label):
        return label[len(self._name) + 1:]

    @property
    def nuisance_parameters(self) -> Dict[str, Parameter]:
        # Currently, the only nuisance parameters are the ones for the
        # bkg We could have systematics here as well
        return self._threeml_bkg_parameters

    def update_nuisance_parameters(self, new_nuisance_parameters: Dict[str, Parameter]):
        # Currently, the only nuisance parameters are the ones for the
        # bkg We could have systematics here as well
        self._threeml_bkg_parameters = new_nuisance_parameters

        # Set underlying bkg model
        self._update_bkg_parameters()

    def _update_bkg_parameters(self, name = None):
        # 1. Remove plugin name. Opposite of the nuisance_parameters
        # property
        # 2. Convert to "bare" Quantity value
        if self._bkg is not None:
            if name is None:
                #Update all
                self._bkg.set_parameters(**{self._remove_prefix_name(label): parameter.as_quantity for label, parameter in self._threeml_bkg_parameters.items()})
            else:
                # Only specific value
                self._bkg.set_parameters(**{name:self._threeml_bkg_parameters[self._add_prefix_name(name)].as_quantity})

    class _Bkg_parameter:
        # Allows idiom plugin.bkg_parameters["bkg_param_name"] to get
        # 3ML parameter
        def __init__(self, plugin):
            self._plugin = plugin
        def __getitem__(self, name):
            # Adds plugin name, required by 3ML code
            return self._plugin._threeml_bkg_parameters[self._plugin._add_prefix_name(name)]
        def __setitem__(self, name, param: Parameter):
            if param.name != self[name].name:
                raise ValueError(f"Name of new set parameter need to match existing parameters ({param.name} != {self[name].name})")
            self._plugin._threeml_bkg_parameters[self._plugin._add_prefix_name(name)] = param
            self._plugin._update_bkg_parameters(name)


    def get_number_of_data_points(self) -> int:
        return self._like.nobservations

    def set_model(self, model):
        self._response.set_model(model)

    def get_log_like(self):
        # Update underlying background object in case the Parameter
        # objects changed internally
        self._update_bkg_parameters()

        return self._like.get_log_like()

    def inner_fit(self):
        """
        Required for 3ML fit.

        Maybe in the future use fast norm fit to minimize the
        background normalization

        """
        return self.get_log_like()
