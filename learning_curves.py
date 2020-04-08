import matplotlib.pyplot as plt

from labellines import labelLines

import math
import numpy as np
import pickle
import os



directory = "results/CURVES/"
folder_list = ["tiles-plain", "tilings-plain", "const-reg", "const-mt"]
label_placements = {}
label_placements["tiles-plain"] = [[0.5, 0.4, 0.3, 0.2], [0.6,0.2, 0.1, 0.35]]
label_placements["tilings-plain"] = [[0.55, 0.12, 0.36, 0.24], [0.3,0.47, 0.3, 0.1]]
label_placements["const-reg"] = [[0.55, 0.12, 0.36, 0.24], [0.55,0.35, 0.45, 0.15]]
label_placements["const-mt"] = [[0.15, 0.38, 0.25, 0.5], [0.35,0.45, 0.40, 0.55]]


labels = ["constant tiles/dimension (10) ", "constant number of tilings (16)", "constant resolution (1600)", "non-uniform grid"]
fig_folder = directory + "figcurves/"
results_dict = {}
row = -1
for folder in folder_list:
    row += 1
    for h in range(2):


        fig, ax = plt.subplots()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        #plt.set_size_inches(8, 11)
        files_list = os.listdir(directory + folder )
        n = -1
        for filename in files_list:
            results_dict[filename] = {}
            results_dict[filename]["x"] = []
            results_dict[filename]["y"] = []
            results_dict[filename]["y_err"] = []
            with open(directory + folder + "/" + filename, "rb") as results_file:
                results = pickle.load(results_file)
                settings = results["experimental_settings"]
                numRuns = settings["numRuns"]
                numEps = settings["numEps"]
                horizons = [[0, 100], [numEps -100, numEps]]
                for item in results["experimental_data"]:
                    algo_name = item["algo"]
                    rawdata = np.array(item["rawdata"])
                    avg_splice = rawdata[:,horizons[h][0] : horizons[h][1]].mean(axis=1)
                    avg_val = avg_splice.mean()
                    std_error = avg_splice.std() / math.sqrt(numRuns)
                    results_dict[filename]["y"].append(avg_val)
                    results_dict[filename]["y_err"].append(std_error)
                    results_dict[filename]["x"].append(item["alpha"] * item["numtilings"])
        for key in files_list:
            x = np.array(results_dict[key]["x"])
            y = np.array(results_dict[key]["y"])
            y_err = np.array(results_dict[key]["y_err"])
            i = np.argsort(x)
            x = x[i]
            y = y[i]
            y_err = y_err[i]
            print(y.sum())
            print(y)
            if y.sum() == 0:
                continue
            print(key)
            _, changed = key.split("_")
            if folder == "const-reg":
                if changed == "8":
                    changed = "8/7.07"
                elif changed == "16":
                    changed = "16/10"
                elif changed == "32":
                    changed = "32/14.14"
                elif changed == "64":
                    changed = "64/5"
            plt.errorbar(x, y, y_err, linewidth=1, label=changed)
            print(key)
            plt.xlabel(r'$\alpha * num tilings$ ')

        #plt.legend(files_list)
        #plt.title(algo_name)
        #axs[row,h].xlabel("Alpha * n (number of tilings)")
        if h == 0:
            #axs[row,h].ylabel("Average steps first 100 episodes over " + str(numRuns)  + " runs")
            plt.axis([0, 1, 100, 500])
            plt.yscale("linear")
            plt.ylabel("Average Steps (first 100 eps)")
            #plt.savefig(fig_folder + folder + "short")
        else:
            #axs[row,h].ylabel("Average steps episodes 900-1000 over " + str(numRuns) + " runs")
            plt.axis([0, 1, 100, 200])
            plt.yscale("linear")
            plt.ylabel("Average Steps (last 100 eps)")
            #plt.savefig(fig_folder + folder + "long")
        plt.tight_layout(2)
        if h == 0:
            suffix = "short"
        else:
            suffix = "long"
        print(fig_folder + folder + suffix)
        labelLines(plt.gca().get_lines(), xvals= label_placements[folder][h])
        plt.savefig(fig_folder + folder + suffix)
        plt.close()