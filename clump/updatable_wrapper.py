def make_object_updatable(cluster_object, updatable_parameters_name_list):
    for par_name in updatable_parameters_name_list:
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
