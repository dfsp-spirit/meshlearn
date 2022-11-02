# -*- coding: utf-8 -*-

# Data post-proc settings. These should be moved into the data loading pipeline.

postproc_settings = {
    "do_replace_nan"       : True,
    "replace_nan_with"     : 0.0,
    "do_scale_descriptors" : True,
    "scale_func"           : lambda x : x - x.min(0) / x.ptp(0)
}