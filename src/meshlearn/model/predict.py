# -*- coding: utf-8 -*-

"""
Model prediction functions.

This file is part of meshlearn, see https://github.com/dfsp-spirit/meshlearn for details.
"""

from abc import abstractmethod
from tabnanny import verbose
import numpy as np
import os
from meshlearn.model.persistance import load_model
import brainload.nitools as nit
import brainload.meshexport as meshexport
import nibabel.freesurfer.io as fsio
from meshlearn.data.training_data import compute_dataset_for_mesh
import time
from datetime import timedelta
from pathlib import Path

class MeshPredict():
    """
    Predict per-vertex descriptor values for a mesh.

    Parameters
    ----------
    mesh_file  : str, path to input mesh file.
    descriptor : str, the desciptor name. Currently only 'pial_lgi' is supported.
    """

    model = None
    model_settings = None
    verbose = True

    @abstractmethod
    def predict(self, mesh_file):
        pass


class MeshPredictLgi(MeshPredict):
    """
    Predict lGI for a mesh file.
    """
    def __init__(self, model_file, model_settings_file, verbose=True):
        self.verbose = verbose
        if isinstance(model_file, str) and isinstance(model_settings_file, str):
            self.model, self.model_settings = load_model(model_file, model_settings_file, verbose=self.verbose)
        elif isinstance(model_file, str) and (model_settings_file is None or isinstance(model_settings_file, dict)):
                self.model, _ = load_model(model_file, model_settings_file=None, verbose=self.verbose)
                self.model_settings = model_settings_file  # Assume it is a dict with the model info (or None).
        if hasattr(model_file, 'predict'):
                self.model = model_file  # Assume it is a model that has an sklearn-like API. It must support `x.precict()`.
        if isinstance(model_settings_file, dict):
            self.model_settings = model_settings_file  # Assume it is a dict with the model settings.

        if not hasattr(self.model, 'predict'):
            raise ValueError("Model obtained based on parameter 'model_file' has no 'model.predict' method. Pass a valid sklearn model or str representing a path to a pkl file containing one.")
        if not isinstance(self.model_settings, dict):
            raise ValueError("Model settings obtained based on parameter 'model_settings_file' is not a dict. Cannot perform pre-processing for the model if we do not have its settings.")
        if not 'data_settings' in self.model_settings:
            raise ValueError("Model settings obtained based on parameter 'model_settings_file' is a dict but is missing key 'data_settings' with data pre-processing settings. Cannot perform pre-processing of data for prediction if we do not have these settings.")

    def _extract_preproc_settings(self, model_settings):
        """
        Extract the relevant pre-processing settings used for model training from the model settings.

        We need to pre-process the model we want to predict on in the same way.
        """
        if 'preproc_settings' in model_settings:  # They are stored in there directly, this is the new version (not legacy).
            return model_settings['preproc_settings']
        else:
            preproc_settings = dict()  # Legacy file format version,
            keys_of_interest = ["add_desc_brain_bbox", "add_desc_neigh_size", "add_desc_vertex_index", "cortex_label", "filter_smaller_neighborhoods", "mesh_neighborhood_count" , "mesh_neighborhood_radius"]
            for k in keys_of_interest:
                if k in model_settings['data_settings']:
                    preproc_settings[k] = model_settings['data_settings'][k]
        return preproc_settings


    def predict(self, mesh_file):
        """
        Predict per-vertex descriptor values for a mesh or a list of meshes.

        Parameters
        ----------
        mesh_file : str or list of str, path(s) to one or more meshes in FreeSurfer surf format (i.e., `lh.pial` or `rh.pial` surface of recon-all output).

        Returns
        -------
        1d np.ndarray of floats, the predicted per-vertex descriptor values. If parameter `mesh_file` is a list, a list of such arrays in returned.
        """
        if self.verbose:
            preproc_start = time.time()

        preproc_settings = self._extract_preproc_settings(self.model_settings)

        if self.verbose:
            print(f"preproc_settings: {preproc_settings}")

        is_list = isinstance(mesh_file, list)
        if not is_list:
            mfl = [mesh_file]
        else:
            mfl = mesh_file

        res_list = []

        for file_idx, meshf in enumerate(mfl):
            dataset, _, _ = compute_dataset_for_mesh(meshf, preproc_settings)

            if self.verbose:
                print(f"Handling mesh file '{meshf}'.")
                preproc_end = time.time()
                preproc_execution_time = preproc_end - preproc_start
                preproc_execution_time_readable = timedelta(seconds=preproc_execution_time)
                print(f" - Pre-processing mesh took {preproc_execution_time_readable}.")
                predict_start = time.time()

            res_list.append(self.model.predict(dataset))

            if self.verbose:
                predict_end = time.time()
                predict_execution_time = predict_end - predict_start
                predict_execution_time_readable = timedelta(seconds=predict_execution_time)
                print(f" - Predicted {res_list[file_idx].size} descriptor values in range {np.min(res_list[file_idx])} to {np.max(res_list[file_idx])}.")
                print(f" - Prediction of {dataset.shape[0]} values took {predict_execution_time_readable}.")

        if is_list:
            return res_list
        else:
            return res_list[0]

    def predict_for_recon_dir(self, recon_dir, subjects_list=None, subjects_file=None, outname="pial_lgi_p", surface="pial", hemis=["lh", "rh"], do_write_files=True, outdir=None, write_ply=False):
        """
        Predict per-vertex descriptor values for all meshes in a `recon-all` output directory structure, and save the resulting descriptors to disk,
        into the existing recon-all output structure.

        Parameters
        ----------
        recon_dir       : str, path to recon-all output directory containing MRI-data pre-processed with FreeSurfer.
        subjects_list   : list of str, the subjects to handle in the subjects dir. Leave at None to use all subjects detected in there. See also `subjects_file`.
        subjects_file   : str, path to subjects file, a txt file containing one subject per line. If this is given (not None), subjects_list must be None.
        outname         : str, the ouput name of the predicted descriptor files. The output format will be FreeSurfer curv format. A prefix based on the hemisphere (one of `lh.` or `rh.`) will be added to construct the full file name. The file will be placed in the `<recon_dir>/<subject>/surf/` directory of each subject.
        surface         : str, the input surface (mesh) to use. Must be 'pial' for pial_lgi, or you will get very bad results. Only in here to be able to re-use this method with other models/descriptors.
        hemis           : list of str, max len 2, entries must be a subset of: `['lh', 'rh']`.
        outdir          : str or None, an alternative output directory. Only needed if you do not want to write to the input directory structure (e.g., because you do not have write access there). Only the base dir must exist, the structure underneath will be created for you. Leave at 'None' to write to 'recon_dir'.
        do_write_files  : bool, whether to save results as curv files to disk.
        write_ply       : bool, whether to also write output as vertex-colored mesh in PLY format. Useful for quick inspection in the external software like 'meshlab'. Ignored if do_write_files is False.

        Returns
        -------
        pvd_files_written   : list of str, the filenames of output files that were written successfully.
        infiles_okay        : list of str, the input filenames for the files in `pvd_files_written`, in the same order.
        infiles_missing     : list of str, the filenames of expected input filenames (meshes) that were not found in the recon-all directory structure.
        infiles_with_errors : list of str, the filenames of files where the input mesh file existed, but an error occurred during prediction or writing the output file.
        values              : list of np.ndarray, the predicted values for the files in `infiles_okay`.
        """
        if not os.path.isdir(recon_dir):
            raise ValueError(f"Input directory '{recon_dir}' does not exist or cannot be read.")
        if subjects_list is not None and subjects_file is not None:
            raise ValueError("Only one of parameters 'subjects_list' and 'subjects_file' can be used.")
        if subjects_list is None and subjects_file is None:
            raise ValueError("Exactly one of parameters 'subjects_list' and 'subjects_file' must be given (and not None).")
        if subjects_file is not None:
            subjects_list = nit.read_subjects_file(subjects_file)
        if not isinstance(subjects_list, list):
            raise ValueError("Parameter 'subjects_list' must be a list of str.")

        if outdir is not None:
            if not os.path.isdir(outdir):
                raise ValueError(f"If parameter 'outdir' is given, that directory must exist, but '{outdir}' cannot be read or is not a directory.")

        pvd_files_written = []
        infiles_okay = []
        infiles_missing = []
        infiles_with_errors = []
        values = []

        for subject in subjects_list:
            subj_surf_dir = os.path.join(recon_dir, subject, 'surf')
            output_surf_dir = subj_surf_dir
            if outdir is not None:
                output_surf_dir = os.path.join(outdir, subject, 'surf')
                Path(output_surf_dir).mkdir(parents = True,  exist_ok = True)
            for hemi in hemis:
                mesh_file = os.path.join(subj_surf_dir, hemi + "." + surface)
                if os.path.isfile(mesh_file):
                    outfile = os.path.join(output_surf_dir, hemi + outname)
                    try:
                        pvd = self.predict(mesh_file)
                        infiles_okay.append(mesh_file)
                        values.append(pvd)

                        if do_write_files:
                            try:
                                fsio.write_morph_data(outfile, pvd)
                                pvd_files_written.append(outfile)
                                if self.verbose:
                                        print(f"Predicted per-vertex descriptor data for mesh '{mesh_file}' written in curv format to '{outfile}'.")
                            except Exception as ex:
                                print(f"Failed to write predicted per-vertex descriptor data output for mesh '{mesh_file}' to '{outfile}': {str(ex)}")

                            if write_ply:
                                try:
                                    vert_coords, faces = fsio.read_geometry(mesh_file)
                                    vertex_colors = meshexport.scalars_to_colors_matplotlib(pvd, "viridis")
                                    ply_str = meshexport.mesh_to_ply(vert_coords, faces, vertex_colors=vertex_colors)
                                    ply_outfile = os.path.join(output_surf_dir, hemi + outname + ".ply")
                                    with open(ply_outfile, "w") as text_file:
                                        text_file.write(ply_str)
                                    if self.verbose:
                                        print(f"Predicted per-vertex descriptor data for mesh '{mesh_file}' written in PLY format to '{ply_outfile}'.")
                                except Exception as ex:
                                    print(f"Failed to write predicted per-vertex descriptor data for mesh '{mesh_file}' in PLY format to '{outfile}': {str(ex)}")

                    except Exception as ex:
                        print(f"Failed to predict per-vertex descriptor data for mesh '{mesh_file}': {str(ex)}")
                        infiles_with_errors.append(mesh_file)

                else:
                    if self.verbose:
                        print(f"Subject {subject} is missing {hemi} input mesh file at '{mesh_file}'. Skipping.")
                    infiles_missing.append(mesh_file)

        return pvd_files_written, infiles_okay, infiles_missing, infiles_with_errors, values




