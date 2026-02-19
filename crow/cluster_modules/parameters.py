"""The parameters module.

This module holds the class that stores and manages parameter to be used in
each cluster_module object.
"""


class Parameters:
    """The parameter class that stores and manages parameter to be used in
    each cluster_module object.
    """

    def __init__(self, default_parameters_dict):
        """Create a Parameters container.

        Parameters
        ----------
        default_parameters_dict : dict
            Dictionary with the default parameter names and values. Only keys
            present in this dictionary will be accepted by this instance.
        """
        self.__pars = {**default_parameters_dict}

    def __getitem__(self, item):
        """Retrieve a parameter value by key.

        Parameters
        ----------
        item : str
            Name of the parameter to retrieve.

        Returns
        -------
        Any
            The parameter value stored under ``item``.

        Raises
        ------
        KeyError
            If ``item`` is not a valid parameter name for this instance.
        """
        return self.__pars[item]

    def __setitem__(self, item, value):
        """Set a parameter value for an existing key.

        Parameters
        ----------
        item : str
            Name of the parameter to set. Must already exist in the container.
        value : Any
            Value to assign to the parameter.

        Raises
        ------
        KeyError
            If ``item`` is not one of the predefined parameter names.
        """
        if item not in self.__pars:
            raise KeyError(
                f"key={item} not accepted, " f"must be in {list(self.__pars.keys())}"
            )
        self.__pars[item] = value

    def keys(self):
        """Return an iterable view of parameter names.

        Returns
        -------
        dict_keys
            A view of the parameter keys in this container.
        """
        return self.__pars.keys()

    def values(self):
        """Return an iterable view of parameter values.

        Returns
        -------
        dict_values
            A view of the parameter values in this container.
        """
        return self.__pars.values()

    def items(self):
        """Return an iterable view of (key, value) pairs.

        Returns
        -------
        dict_items
            A view of the parameter (key, value) pairs in this container.
        """
        return self.__pars.items()

    def __iter__(self):
        """Iterate over parameter keys.

        Yields
        ------
        str
            Parameter key names.
        """
        for key in self.keys():
            yield key

    def update(self, update_dict):
        """Update multiple parameters at once.

        Parameters
        ----------
        update_dict : dict or Parameters
            Mapping of parameter names to new values. Keys must be a subset of
            the parameters defined for this container.

        Raises
        ------
        ValueError
            If ``update_dict`` is not a dict or Parameters instance.
        KeyError
            If ``update_dict`` contains keys not defined in this container.
        """
        if not isinstance(update_dict, (dict, Parameters)):
            raise ValueError(
                "argument of update must be dict or Parameters, "
                f"{type(update_dict)} given!"
            )
        bad_keys = list(filter(lambda key: key not in self.__pars.keys(), update_dict))
        if len(bad_keys) > 0:
            raise KeyError(f"bad keys provided for update: {bad_keys}")
        self.__pars.update(update_dict)
