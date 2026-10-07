import os
import pm4py
from config_loader import PROJECT_ROOT

petri_nets_dir = os.path.join(PROJECT_ROOT, "models/petri_nets")
os.makedirs(petri_nets_dir, exist_ok=True)

# Load and visualize all Petri net models found in models/petri_nets
pnml_files = [f for f in sorted(os.listdir(petri_nets_dir)) if f.endswith(".pnml")]

if not pnml_files:
    print(f"No .pnml files found in {petri_nets_dir}")
else:
    for fn in pnml_files:
        pnml_path = os.path.join(petri_nets_dir, fn)
        out_png = os.path.join(petri_nets_dir, fn.replace(".pnml", "_visual.png"))
        try:
            net, im, fm = pm4py.read_pnml(pnml_path)
            pm4py.save_vis_petri_net(net, im, fm, out_png)
            print(f"[✓] Rendered {fn} -> {os.path.basename(out_png)}")
        except Exception as e:
            print(f"[!] Warning: Could not render {fn}: {e}")