from firecrown.updatable import Updatable

class UpdatableClusterObjects(Updatable):
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
    def __init__(self, cluster_objects_list, cluster_objects_updatable_parameters):
        super().__init__()
        self.cluster_objects_list = cluster_objects_list
        self.cluster_objects_updatable_parameters = cluster_objects_updatable_parameters
        for cl_obj, updatable_list in zip(cluster_objects_list, cluster_objects_updatable_parameters):
            self._make_object_updatable(cl_obj, updatable_list)
    def _make_object_updatable(cluster_object, updatable_parameters):
        for par_name in updatable_parameters:
            setattr(cluster_object, par_name) = parameters.register_new_updatable_parameter(
                default_value=getattr(cluster_object, par_name)
            )
# EXAMPLES 
# 
#  MurataBinned:
#      updatable_parameters_name_list:
#         mu_p0, default_value, default_value, default_value, default_value, default_value
# 
#  MurataUnbinned:
#      updatable_parameters_name_list:
#         mu_p0, default_value, default_value, default_value, default_value, default_value
# 
#  Completeness:
#      updatable_parameters_name_list:
#         ac_nc, default_value, default_value, default_value
# 
#  Purity:
#      updatable_parameters_name_list:
#         ap_nc, default_value, default_value, default_value
# 
#  ClusterDeltaSigma:
#      updatable_parameters_name_list:
#         cluster_conc 
