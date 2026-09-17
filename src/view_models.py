import pm4py

#load Petri net
net_split, im_split, fm_split = pm4py.read_pnml("../models/discovered_model_split.pnml")
net_ind, im_ind, fm_ind = pm4py.read_pnml("../models/discovered_model_inductive.pnml")

#save it as a png
pm4py.save_vis_petri_net(net_split, im_split, fm_split, "../models/split_miner_visual.png")
print("Saved Split Miner image")

pm4py.save_vis_petri_net(net_ind, im_ind, fm_ind, "../models/inductive_miner_visual.png")
print("Saved Inductive Miner image")