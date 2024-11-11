import pcdl
import glob
import xml.etree.ElementTree as ET

def load_simulation(sim_out_dir, time_step = 1440, **kwargs):
    all_timesteps = [pcdl.pyMCDS(ts, settingxml='config.xml') for ts in sorted(glob.glob(sim_out_dir+"output*.xml"))]
    selected_steps = [ts for ts in all_timesteps if ts.get_time() % time_step == 0]
    return selected_steps

def load_simulation_by_interval(sim_out_dir, interval = 1440, **kwargs):
    print("Loading simulations...")
    tree = ET.parse(sim_out_dir+'config.xml')
    xml_root = tree.getroot()
    save_interval = int(xml_root.find('.//save/full_data/interval').text)
    print(f"Save interval: {save_interval}")
    if save_interval < interval:
        all_output_files = sorted(glob.glob(sim_out_dir+"output*.xml"))
        selected_output_files = all_output_files[0::int(interval/save_interval)]
        selected_steps = [pcdl.pyMCDS(ts, settingxml='config.xml') for ts in selected_output_files]
        return selected_steps
    else:
        print("Provided interval must be larger than the save interval.")
