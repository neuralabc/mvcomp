from .mvcomp import mysort, feature_gen, norm_covar_inv, mah_dist_feat_mat, mah_dist_mat_2_roi, subject_list, feature_list, compute_average, model_comp, spatial_mvcomp, dist_plot, correlation_fig
from .mvcomp import model_comp_simplified, compute_average_simplified
from .version import __version__

# initial hack to not import optional plotting functions if necessary packages do not exist
try:
    from .plotting import *
except ImportError:
    print(ImportError)