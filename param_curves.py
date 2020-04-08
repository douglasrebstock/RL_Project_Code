import matplotlib.pyplot as plt

import math
import numpy as np
import pickle
import os

from labellines import labelLines

files_list = []
results_keys = [5, 10] #, 15, 20]
results_dict = {}
algo_name = ""
numRuns = 100


directory = "results/CURVES/"
folder_list = ["const-mt","const-reg","tiles-plain","tilings-plain"]
offset_dict = {}
offset_dict["const-mt"] = [-20, 20]
offset_dict["const-reg"] = [20, -15]
offset_dict["tiles-plain"] = [-22, 15]
offset_dict["tilings-plain"] = [20, -15]
label_placements = {}
label_placements["tiles-plain"] = [750, 750]
label_placements["tilings-plain"] = [750, 750]
label_placements["const-reg"] = [750, 750]
label_placements["const-mt"] = [750, 750]


fig_folder = directory + "figcurves/"
best = {}
best["tiles-plain"] = [0.5, 0.3, 0.4, 0.2]
best["tilings-plain"] = [0.3, 0.3, 0.3, 0.3]
best["const-mt"] = [0.2, 0.4, 0.2, 0.2]
best["const-reg"] = [0.2, 0.3, 0.4, 0.3]

best["tiles-plain"] = [0, 0, 0, 0]
best["tilings-plain"] = [0, 0, 0, 0]
best["const-mt"] = [0, 0, 0, 0]
best["const-reg"] = [0, 0, 0, 0]

best["tiles-plain"] = [0.5, 0.0, 0.0, 0.2]
best["tilings-plain"] = [0.3, 0.0, 0.3, 0.0]
best["const-mt"] = [0.2, 0.0, 0.0, 0.2]
best["const-reg"] = [0.2, 0.0, 0.0, 0.3]

for folderhack in folder_list:
    legend = []
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for folder in folder_list:
        files_list = os.listdir(directory + folder )
        n = -1
        for filename in files_list:
            n += 1
            results_dict[filename] = {}
            #results_dict[filename]["avg_curve"] = []
            #results_dict[filename]["std"] = []
            with open(directory + folder + "/" + filename, "rb") as results_file:
                results = pickle.load(results_file)
                settings = results["experimental_settings"]
                numRuns = settings["numRuns"]
                numEps = settings["numEps"]
                for item in results["experimental_data"]:
                    algo_name = item["algo"]
                    if item["alpha"] * item["numtilings"] != best[folder][n] or folderhack != folder:
                        continue
                    #awdata = np.array(item["rawdata"])
                    x = np.arange(numEps)
                    std_error = np.array(item["std"]) / math.sqrt(numRuns)
                    avg_steps = np.array(item["avg"])
                    if "const" not in folderhack:
                        changed = "[" + str(item["alpha"] * item["numtilings"]) + "," + str(item["numTiles"]) + "," + str(item["numtilings"]) + "]"
                    else:
                        changed = "[" + str(item["alpha"] * item["numtilings"]) + "," + str(1600) + "," + str(item["numtilings"]) + "]"

                    plt.errorbar(x, avg_steps, std_error, linewidth=0.5, label=changed)
                    legend.append(algo_name + folder + str(item["alpha"] * item["numtilings"]) + str(
                        item["numtilings"] * item["numTiles"] * item["numTiles"]) + str(item["numtilings"]))
    plt.ylabel("Average Step per Epiode (over 100 runs)")
    plt.xlabel("Episode Number")
    plt.axis([0, numEps, 80, 500])
    plt.yscale("log")
    #plt.legend(legend)
    labelLines(plt.gca().get_lines(), offsets=offset_dict[folderhack], align=False, xvals=label_placements[folderhack])
    plt.savefig(fig_folder + folderhack + "curve")
    plt.close()

