"""The parameters module.

This module holds the class that stores and manages parameter to be used in
each cluster_module object.
"""


class Parameters:
    """The parameter class that stores and manages parameter to be used in
    each cluster_module object.
    """

    def __init__(self, default_parameters_dict):
        """
        Parameters
        ----------
        default_parameters_dict: dict
            Dictionary with the default parameters names and values.
            Only parameters defined in this dictionary will be accepted
            in this class
        """
        self.__pars = {**default_parameters_dict}

    def __getitem__(self, item):
        return self.__pars[item]

    def __setitem__(self, item, value):
        if item not in self.__pars:
            raise KeyError(
                f"key={item} not accepted, " f"must be in {list(self.__pars.keys())}"
            )
        self.__pars[item] = value

    def keys(self):
        return self.__pars.keys()

    def values(self):
        return self.__pars.values()

    def items(self):
        return self.__pars.items()

    def __iter__(self):
        for key in self.keys():
            yield key

    def update(self, update_dict):
        if not isinstance(update_dict, (dict, Parameters)):
            raise ValueError(
                "argument of update must be dict or Parameters, "
                f"{type(update_dict)} given!"
            )
        bad_keys = list(filter(lambda key: key not in self.__pars.keys(), update_dict))
        if len(bad_keys) > 0:
            raise KeyError(f"bad keys provided for update: {bad_keys}")
        self.__pars.update(update_dict)
