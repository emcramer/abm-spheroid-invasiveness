import numpy as np
import pandas as pd
import skimage as ski
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import cv2, pcdl, glob, os, time, math

# custom functions
import image_processing as ip
import invasiveness as iv
import pseudo_image_utils as piu
import utils

hypo2_physicell_output_directories = "../PhysiCell/output/hypothesis2_ics_permuted/"
hypo2_out_dirs = sorted(
    [d for d in os.listdir(hypo2_physicell_output_directories) if os.path.isdir(hypo2_physicell_output_directories+d)]
)

# only run if the results file needs to be re-generated
hypo2_simulation_results = []
hypo2_sim_props = {}
for j, d in enumerate(tqdm(hypo2_out_dirs)):
    print(f"Loading simulation: {d}")
    selected_steps = utils.load_simulation_by_interval("../PhysiCell/output/hypothesis2_ics_permuted/"+d+"/")
    print("Finished loading.")
    n_steps = len(selected_steps)
    timesteps = [f"{24*k}h" for k in range(n_steps)]
    invasive_area_ratios = [None] * n_steps
    spheroid_core_areas = [None] * n_steps
    invasive_areas = [None] * n_steps
    invasive_projection_heights = [None] * n_steps
    n_invasive_projections = [None] * n_steps
    
    print(f"Processing simulation: {d}")
    rad_threshold = 0
    for i, ts in enumerate(selected_steps):
        pseudo_brightfield, cells, substrates = piu.abm2gray(ts,substrates=['ecm'])
        contour, matrices = iv.calc_contour(pseudo_brightfield)
        if i == 0: # if this is the first time step, then set the threshold based on the median radial distance
            invasive_projection_heights[i] = iv.invasive_projection_height(contour)
            rad_threshold = np.median(invasive_projection_heights[i])
        else:
            invasive_projection_heights[i] = iv.invasive_projection_height(contour, median_radial_distance=rad_threshold)
        invasive_area_ratios[i], spheroid_core_areas[i], invasive_areas[i] = iv.invasive_area_ratio(contour, median_radial_distance=rad_threshold)
        n_invasive_projections[i] = iv.count_invasive_projections(contour, median_radial_distance=rad_threshold)

    props_df = pd.DataFrame({
    'sample_name': f"sample_{j}",
    'experimental_condition': d[0:-4],
    'timepoint':timesteps,
    'n_invasive_projections':n_invasive_projections,
    'invasive_projection_heights': invasive_projection_heights,
    'invasive_areas':invasive_areas,
    'spheroid_core_areas':spheroid_core_areas,
    'invasive_area_ratios':invasive_area_ratios
    })
    props_df.set_index('timepoint', inplace=True)
    hypo2_simulation_results.append(props_df)
    print("Finished processing simulation.")
    
hypo2_simulation_results_df = pd.concat(hypo2_simulation_results)
hypo2_simulation_results_df['hours'] = [int(hr[0:-1]) for hr in hypo2_simulation_results_df.index]
hypo2_simulation_results_df['experimental_condition'] = ["_".join(s.split("_")[0:-1]) for s in hypo2_simulation_results_df['experimental_condition']]
hypo2_simulation_results_df.to_csv("../output/tables/hypothesis1_invasiveness_random_ics_2.csv")

