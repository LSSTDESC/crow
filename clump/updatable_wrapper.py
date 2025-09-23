from firecrown.updatable import Updatable
from firecrown import parameters


class UpdatableParameters(Updatable):
    """
    Class that contains all cluster objects to integrate the firecrown
    Updatable functionality.

    Parameters
    ----------
    cluster_objects_list: list
        List of all cluster objects.
    cluster_objects_updatable_parameters: list
        List of list of the names of updatables in each cluster object.
        Same order as cluster_objects_list.
    """

    def __init__(self, updatable_parameters):
        super().__init__()
        self.updatable_parameters = updatable_parameters

    def import_parameters(self, cluster_object):
        for par_name in self.updatable_parameters:
            setattr(
                self,
                par_name,
                parameters.register_new_updatable_parameter(
                    default_value=getattr(cluster_object, par_name)
                ),
            )

    def export_parameters(self, cluster_object):
        for par_name in self.updatable_parameters:
            setattr(
                cluster_object,
                par_name,
                getattr(self, par_name),
            )


class UpdatableClusterObjects:
    """
    Class that contains all cluster objects to integrate the firecrown
    Updatable functionality.

    Parameters
    ----------
    cluster_objects_list: list
        List of all cluster objects.
    cluster_objects_updatable_parameters: list
        List of list of the names of updatables in each cluster object.
        Same order as cluster_objects_list.
    """

    def __init__(self, cluster_objects_names, cluster_objects_updatable_parameters):
        self.cluster_objects_names = cluster_objects_names
        for name, par_names in zip(cluster_objects_names, cluster_objects_updatable_parameters):
            setattr(
                self,
                name,
                UpdatableParameters(par_names),
            )

    def import_parameters(self, cluster_recipe):
        for name in self.cluster_objects_names:
            getattr(self, name).import_parameters(getattr(cluster_recipe, name))

    def export_parameters(self, cluster_recipe):
        for name in self.cluster_objects_names:
            getattr(self, name).export_parameters(getattr(cluster_recipe, name))


# EXAMPLES
#
#  MurataBinned:
#      updatable_parameters_name_list:
#         mu_p0,  mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2
#
#  MurataUnbinned:
#      updatable_parameters_name_list:
#         mu_p0,  mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2
#
#  Completeness:
#      updatable_parameters_name_list:
#         ac_nc, bc_nc, ac_rc, bc_rc 
#
#  Purity:
#      updatable_parameters_name_list:
#         ap_nc, bp_nc, ap_rc, bp_rc 
#
#  ClusterAbundance:
#      updatable_parameters_name_list:
#          cosmo
#
#  ClusterDeltaSigma:
#      updatable_parameters_name_list:
#         cosmo, cluster_conc
