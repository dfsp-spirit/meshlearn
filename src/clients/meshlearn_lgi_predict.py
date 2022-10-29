#!/usr/bin/env python


import argparse
import os
from meshlearn.model.predict import MeshPredictLgi
import brainload.nitools as nit
import brainload.meshexport as meshexport
import nibabel.freesurfer.io as fsio


def predict_lgi():

    example_text = '''Examples:

 # Predict for a single mesh file, in verbose mode, using the model and implicitely given json file .../ml_model.json:
 meshlearn_lgi_predict -p $SUBJECTS_DIR/subject1/surf/lh.pial -v tests/test_data/models/lgbm_lgi/ml_model.pkl
 # Predict for all mesh files in recon-all output dir, using the model and explicitely given json file:
 meshlearn_lgi_predict -r $SUBJECTS_DIR -j ./ml_model_metadata.json tests/test_data/models/lgbm_lgi/ml_model.pkl
 '''

    parser = argparse.ArgumentParser(prog='meshlarn_lgi_predict',
                                     description="Use trained model to predict per-vertex lGI descriptor for a brain mesh.",
                                     epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter
                                     )
    parser.add_argument("model", help="str, the model file containg the model to load and to use for predictions. Must be a pickle file (.pkl file extension). Will search for a matching metadata JSON file in the same directory, with file extension '.json', unless -j is given.")
    parser.add_argument("-j", "--json_metadata", help="str, optional. The model-metadata JSON file. If omitted, the filename will be auto-determined based on the 'model' parameter, as described there.", default="")
    predict_group = parser.add_mutually_exclusive_group(required=True)
    predict_group.add_argument("-p", "--predict-file", help="List of mesh file(s) to predict lGI for. They can be in different dirs, but their names should start with '.lh' or '.rh'. The file names must have at least 3 letters, and the first letters must differ for files in the same directory, as these are used as the prefix for the outfile (see -o).", nargs="+")
    predict_group.add_argument("-r", "--predict-recon-dir", help="List of 'recon-all' directory/directories to predict lGI for.", nargs="+")
    parser.add_argument("-o", "--outfile-suffix", help="str, the output suffix for the predicted descriptor files. The output format will be FreeSurfer curv format. A prefix based on the hemisphere (one of `lh.` or `rh.`) will be added to construct the full file name (in '-r' mode this is know, in '-p' mode the first 3 chars of the respective input file are used). Note that this must not include any directories, see '--outdir' to control that.", default="pial_lgi_p")
    parser.add_argument("-d", "--outdir", help="str, alternate output directory. If omitted, with '-r', output will be written into the input directories, in the (sub)-directories where the respective input mesh file is located. If given, it must exist. When used with -r and several recon-dirs, the output of all of them is written to this single directory.", default="")
    parser.add_argument("-l", "--ply", help="Whether to write predicted PVD data additionally as vertex-colored PLY format meshes. Filename will be the one of the output curv file, but with additional '.ply' suffix.", action="store_true")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    args = parser.parse_args()


    model_pkl_file = args.model
    if not os.path.isfile(model_pkl_file):
        raise ValueError(f"Model file '{model_pkl_file}' does not exist or cannot be read.")
    if len(args.json_metadata) > 0:
        metadata_json_file = args.json_metadata
    else:
        model_filepath_noext = os.path.splitext(model_pkl_file)[0]
        metadata_json_file = model_filepath_noext + ".json"
    if not os.path.isfile(metadata_json_file):
        raise ValueError(f"Model JSON metadata file '{metadata_json_file}' does not exist or cannot be read.")

    if len(args.outdir) > 0:
        if not os.path.isdir(args.outdir):
            raise ValueError(f"If parameter 'outdir' is given, the directory must exist, but '{args.outdir}' cannot be read.")

    Mp = MeshPredictLgi(model_pkl_file, metadata_json_file)

    if args.predict_file:
        print(f"Predicting for files: {args.predict_file}")
        if not isinstance(args.predict_file, list):
            args.predict_file = [args.predict_file]
        outdirs = [os.path.dirname(pf) for pf in args.predict_file]
        if len(args.outdir) > 0:
            outdirs = [args.outdir for _ in args.predict_file]

        for mesh_file_idx, mesh_file in enumerate(args.predict_file):
            lgi_predicted = Mp.predict(mesh_file)

            try:
                outfile = os.path.join(outdirs[mesh_file_idx], os.path.basename(mesh_file)[0:3] + args.outfile_suffix)
            except Exception as ex:
                print(f"ERROR: Could not construct outfile name from input filename '{mesh_file}'. The basename '{os.path.basename(mesh_file)}' must contain at least 3 chars (and should start with a prefix 'lh.' or 'rh.').")
                continue
            try:
                fsio.write_morph_data(outfile, lgi_predicted)
                if args.verbose:
                    print(f"Predicted values for input file '{mesh_file}' written to output file '{outfile}'.")
            except Exception as ex:
                print(f"ERROR: Failed to write morph data for input file '{mesh_file}' to output file '{outfile}': {str(ex)}")

            if args.ply:
                try:
                    vert_coords, faces = fsio.read_geometry(mesh_file)
                    vertex_colors = meshexport.scalars_to_colors_matplotlib(lgi_predicted, "viridis")
                    ply_str = meshexport.mesh_to_ply(vert_coords, faces, vertex_colors=vertex_colors)
                    ply_outfile = outfile + ".ply"
                    with open(ply_outfile, "w") as text_file:
                        text_file.write(ply_str)
                    if args.verbose:
                        print(f"Predicted per-vertex descriptor data for mesh '{mesh_file}' written in PLY format to '{ply_outfile}'.")
                except Exception as ex:
                    print(f"Failed to write predicted per-vertex descriptor data for mesh '{mesh_file}' in PLY format to '{ply_outfile}': {str(ex)}")

    if args.predict_recon_dir:
        print(f"Predicting for recon-all dirs: {args.predict_recon_dir}")
        hemis = ["lh", "rh"]
        for recon_dir in args.predict_recon_dir:
            subjects_list = nit.detect_subjects_in_directory(args.predict_recon_dir, ignore_dir_names=["fsaverage"], required_subdirs_for_hits=["surf"])
            outdir = None
            if len(args.outdir) > 0:
                outdir = args.outdir
            pvd_files_written, infiles_okay, infiles_missing, infiles_with_errors, _ = Mp.predict_for_recon_dir(recon_dir, subjects_list=subjects_list, hemis=hemis, do_write_files=True, outdir=outdir, outname=args.outfile_suffix)
            if len(infiles_missing):
                print(f"Recon dir '{recon_dir}': The following {len(infiles_missing)} expected input files could not be read: {infiles_missing}")
            if len(infiles_with_errors):
                print(f"Recon dir '{recon_dir}': The following {len(infiles_with_errors)} expected input files resulted in errors: {infiles_with_errors}")
            if args.verbose:
                print(f"Recon dir '{recon_dir}': The following {len(pvd_files_written)} outpuf files were successfully written: {pvd_files_written}")



if __name__ == "__main__":
    predict_lgi()