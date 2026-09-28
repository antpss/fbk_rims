import os
import pm4py
from config_loader import PROJECT_ROOT

petri_nets_dir = os.path.join(PROJECT_ROOT, "models/petri_nets")
os.makedirs(petri_nets_dir, exist_ok=True)

#load Petri nets
split_pnml = os.path.join(petri_nets_dir, "discovered_model_split.pnml")
ind_pnml = os.path.join(petri_nets_dir, "discovered_model_inductive.pnml")

#save png
if os.path.exists(split_pnml):
    net_split, im_split, fm_split = pm4py.read_pnml(split_pnml)
    pm4py.save_vis_petri_net(net_split, im_split, fm_split, os.path.join(petri_nets_dir, "split_miner_visual.png"))
    print("Saved Split Miner image to models/petri_nets/split_miner_visual.png")

if os.path.exists(ind_pnml):
    net_ind, im_ind, fm_ind = pm4py.read_pnml(ind_pnml)
    pm4py.save_vis_petri_net(net_ind, im_ind, fm_ind, os.path.join(petri_nets_dir, "inductive_miner_visual.png"))
    print("Saved Inductive Miner image to models/petri_nets/inductive_miner_visual.png")