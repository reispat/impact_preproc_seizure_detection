
# %% Helper functions for "The impact of EEG preprocessing parameters on ultra-low-power seizure detection"
# imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% plotting function for metrics & energy consumption split by conditions

def plot_condition(df_folds_avg, metrics, condition, save=False):

    # prepare figure
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))

    for ax, metric in zip(axes, metrics):
        
        # get df
        df_plot = df_folds_avg.copy()
        df_plot[condition] = df_plot[condition].astype(int).astype("string")
        
        # drop na (plotting otherwise not possible)
        df_plot.dropna(subset=[metric], inplace=True)
        
        # define order
        if condition == "sampling_rate":
            order=["256", "128", "64"]
        elif condition == "window_length" or condition == "window_length_overlap":
            order=["1", "2", "4", "8"]
        elif condition == "bit_width":
            order=["16", "14", "12", "10", "8"]
        elif condition == "n_channels":
            order=["4", "3", "2", "1"]
            
        # plot
        sns.stripplot(df_plot, x=condition, y=metric, hue=condition, size=3,
                    alpha=0.25, legend=None, palette='dark:#939597', ax=ax, order=order)
        
        sns.boxplot(df_plot, x=condition, y=metric, hue=condition, palette="mako", 
                    showfliers=False, ax=ax, medianprops={"color": "k", "linewidth": 1.5})
            
        # set titles and labels
        if condition == "sampling_rate":
            ax.set_xlabel("Sampling rate (Hz)")
            title = "Influence of sampling rate"

        elif condition == "window_length" and any(df_plot["model_type"].str.contains("_0.5")) == False:
            ax.set_xlabel("Window size (s)")
            title = "Influence of window size (0% overlap)"

        elif any(df_plot["model_type"].str.contains("_0.5")):
            ax.set_xlabel("Window size (s)")
            title = "Influence of window size (50% overlap)"        
            
        elif condition == "bit_width":
            ax.set_xlabel("Digital resolution (bits)")
            title = "Influence of digital resolution"

        elif condition == "n_channels":
            ax.set_xlabel("Number of channels")
            title = "Influence of number of channels"
            
        if metric == "event_sensitivity":      
            ax.set_title("Sensitivity", pad=15)
            ax.set_ylabel("%")
            
            # add significance bars for main effects
            if condition == "sampling_rate":
                
                ax.set_ylim(-2, 115)

                # 256 vs 128
                ax.axhline(y=105, xmin=0.18, xmax=0.5, color='black', linestyle="solid", linewidth=2)
                ax.text(x=0.4, y=103, s="**")
                
                # 256 vs 64
                ax.axhline(y=112, xmin=0.18, xmax=0.83, color='black', linestyle="solid", linewidth=2)
                ax.text(x=0.9, y=110, s="*")
                
            elif condition == "window_length" and any(df_plot["model_type"].str.contains("_0.5")) == False:
                
                ax.set_ylim(-2, 115)

                if any(df_plot["model_type"].str.contains("_0.5")) == True:
                    ax.set_ylim(-2, 140)
                    ax.set_yticks(np.arange(0, 100+20, 20))
                
                # 1 vs 8
                ax.axhline(y=105, xmin=0.12, xmax=0.9, color='black', linestyle="solid", linewidth=2)
                ax.text(x=1.4, y=103, s="*")

            elif condition == "window_length" and any(df_plot["model_type"].str.contains("_0.5")) == True:

                ax.set_ylim(-2, 115)

                # 1 vs 4
                ax.axhline(y=105, xmin=0.12, xmax=0.62, color='black', linestyle="solid", linewidth=2)
                ax.text(x=0.9, y=103, s="*")
                
                # 1 vs 8
                ax.axhline(y=112, xmin=0.12, xmax=0.92, color='black', linestyle="solid", linewidth=2)
                ax.text(x=1.4, y=110, s="**")
                

            elif condition == "n_channels":
                ax.set_ylim(-2, 115)

                # 4 vs 2
                ax.axhline(y=105, xmin=0.12, xmax=0.62, color='black', linestyle="solid", linewidth=2)
                ax.text(x=0.9, y=103, s="***")

                # 4 vs 1
                ax.axhline(y=112, xmin=0.11, xmax=0.88, color='black', linestyle="solid", linewidth=2)
                ax.text(x=1.25, y=110, s="***")


            elif condition == "bit_width":
                ax.set_ylim(-2, 115)

                # 4 vs 1
                ax.axhline(y=112, xmin=0.18, xmax=0.83, color='black', linestyle="solid", linewidth=2)
                ax.text(x=1.8, y=110, s="**")

            
        elif metric == "event_false_detections_per_hour":

            ax.set_title("False detections per hour", pad=15)
            ax.set_ylabel("False detections per hour")
            
            # add significance bars for main effects
            if condition == "sampling_rate":
                ax.set_ylim(-0.2, 16)
                ax.set_yticks(np.arange(0, 14+2, 2))

                # # 256 vs 64
                ax.axhline(y=15.5, xmin=0.18, xmax=0.84, color='black', linestyle="solid", linewidth=2)
                ax.text(x=0.9, y=15.5, s="**")
                

            if condition == "window_length":

                ax.set_ylim(-0.2, 18)
                ax.set_yticks(np.arange(0, 14+2, 2))

                # 1 vs 8
                ax.axhline(y=16.6, xmin=0.12, xmax=0.9, color='black', linestyle="solid", linewidth=2)
                ax.text(x=1.4, y=16.3, s="***")
                
                # # 1 vs 4
                ax.axhline(y=15.5, xmin=0.12, xmax=0.7, color='black', linestyle="solid", linewidth=2)
                ax.text(x=0.9, y=15.3, s="***")
                
                # 1 vs 2
                ax.axhline(y=14.5, xmin=0.12, xmax=0.4, color='black', linestyle="solid", linewidth=2)
                ax.text(x=0.4, y=14.2, s="***")                

            if condition == "n_channels":

                ax.set_ylim(-0.2, 18)
                ax.set_yticks(np.arange(0, 14+2, 2))

                # 4 vs 1
                ax.axhline(y=16.6, xmin=0.12, xmax=0.9, color='black', linestyle="solid", linewidth=2)
                ax.text(x=1.5, y=16.25, s="*")
                
                # # 4 vs 2
                ax.axhline(y=15.5, xmin=0.12, xmax=0.7, color='black', linestyle="solid", linewidth=2)
                ax.text(x=1, y=15.25, s="*")
                
                # 4 vs 3
                ax.axhline(y=14.5, xmin=0.12, xmax=0.4, color='black', linestyle="solid", linewidth=2)
                ax.text(x=0.4, y=14.15, s="*")                
                  

            if condition == "bit_width":
                ax.set_ylim(-0.2, 16)
                ax.set_yticks(np.arange(0, 14+2, 2))           
                
        elif metric == "event_average_detection_delay":

            ax.set_title("Average detection delay", pad=15)
            ax.set_ylabel("Detection delay (s)")
            ax.set_ylim(-35, 130)
            ax.set_yticks(np.arange(-30, 100+30, 30))
            
            if condition == "sampling_rate":
                ax.set_ylim(-35, 120)
                ax.set_yticks(np.arange(-30, 100+30, 30))
                
            elif condition == "window_length":

                if any(df_plot["model_type"].str.contains("_0.5")) == True:
                    ax.set_ylim(-35, 150)
                    ax.set_yticks(np.arange(-30, 100+30, 30))

                    # 1 vs 8
                    ax.axhline(y=120, xmin=0.12, xmax=0.9, color='black', linestyle="solid", linewidth=2)
                    ax.text(x=1.4, y=117, s="***")
                    
                    # # 1 vs 4
                    ax.axhline(y=110, xmin=0.12, xmax=0.7, color='black', linestyle="solid", linewidth=2)
                    ax.text(x=0.9, y=107, s="***")
                    
                    # 1 vs 2
                    ax.axhline(y=100, xmin=0.12, xmax=0.4, color='black', linestyle="solid", linewidth=2)
                    ax.text(x=0.4, y=97, s="*")

                else:
                    ax.set_ylim(-35, 200)
                    ax.set_yticks(np.arange(-30, 100+60, 30))

                    # 1 vs 8
                    ax.axhline(y=180, xmin=0.12, xmax=0.9, color='black', linestyle="solid", linewidth=2)
                    ax.text(x=1.4, y=178, s="**")
                    
                    # # 1 vs 4
                    ax.axhline(y=165, xmin=0.12, xmax=0.7, color='black', linestyle="solid", linewidth=2)
                    ax.text(x=0.9, y=163, s="**")
                
            elif condition == "bit_width": 
                    
                    ax.set_ylim(-35, 150)
                    ax.set_yticks(np.arange(-30, 100+60, 30))
                    
                    # # 16 vs 14 bits
                    ax.axhline(y=135, xmin=0.12, xmax=0.3, color='black', linestyle="solid", linewidth=2)
                    ax.text(x=0.45, y=133, s="*")

            elif condition == "n_channels":
                    ax.set_ylim(-35, 200)
                    ax.set_yticks(np.arange(-30, 100+60, 30))

                    # 4 vs 1
                    ax.axhline(y=180, xmin=0.12, xmax=0.9, color='black', linestyle="solid", linewidth=2)
                    ax.text(x=1.5, y=177, s="*")
                    
                    # 4 vs 2
                    ax.axhline(y=165, xmin=0.12, xmax=0.7, color='black', linestyle="solid", linewidth=2)
                    ax.text(x=1, y=162, s="*")

    # plot energy consumption
    df_energy = pd.read_excel("./data/energy_measurements.xlsx") # get energy consumption data

    # filter data
    if condition == "sampling_rate":
        df_energy = df_energy.query('model.str.contains("baseline")')
    elif condition == "window_length" and any(df_plot["model_type"].str.contains("_0.5")) == False:
        df_energy = df_energy.query('model.str.contains("_0.0") | model.str.contains("256")')
    elif condition == "window_length" and any(df_plot["model_type"].str.contains("_0.5")) == True:
        df_energy = df_energy.query('model.str.contains("_0.5")')    
    elif condition == "bit_width":
        df_energy = df_energy.query('model.str.contains("bit_width") | model.str.contains("256")')
    elif condition == "n_channels":
        df_energy = df_energy.query('model.str.contains("channels") | model.str.contains("256")')
        first_row = df_energy.iloc[[0]]
        df_energy = df_energy.drop(index=0)
        df_energy = pd.concat([df_energy.iloc[:-1], first_row, df_energy.iloc[-1:]], ignore_index=True)

    # plot energy consumption
    ax = sns.pointplot(df_energy, x="model", y="cnn_energy_sec_mean", label="CNN energy", ax=axes[3], color="#469a81")
    sns.pointplot(df_energy, x="model", y="adc_energy_sec_median", label="ADC energy", ax=axes[3], color="#545479")
    ax.set_xticks(list(range(df_folds_avg[condition].nunique())))

    if condition == "sampling_rate":
        ax.set_xticklabels(["256", "128", "64"])
        ax.set_xlabel("Sampling rate (Hz)")

    elif condition == "window_length":
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_xlabel("Window size (s)")

        if condition == "window_length" and any(df_plot["model_type"].str.contains("_0.5")) == False:    
            ax.set_ylim(-5, 100)   
        else:
            ax.set_ylim(-20, 200)

    elif condition == "bit_width":
        ax.set_xticklabels(["16", "14", "12", "10", "8"])
        ax.set_xlabel("Digital resolution (bits)")

    elif condition == "n_channels":
        ax.set_xticklabels(["4", "3", "2", "1"])
        ax.set_xlabel("Number of channels")

    ax.set_ylabel("Energy (µJ/s)")
    ax.set_title("Energy consumption", pad=15)

    # custom error bars
    ax.errorbar(df_energy['model'], df_energy['adc_energy_sec_median'],
        yerr=[df_energy['adc_energy_sec_median'] - df_energy['adc_energy_sec_25pct'], 
            df_energy['adc_energy_sec_75pct'] - df_energy['adc_energy_sec_median']],
        fmt='.', capsize=5, capthick=1, ecolor="#545479", markersize=0)
    plt.legend(frameon=False, fontsize=12)

    # adjust layout
    plt.tight_layout()
    plt.suptitle(title, y=1.05, x=0.52)
    sns.despine()


    # save the plot as svg
    if save:
        output_path = "./"
        os.makedirs(output_path, exist_ok=True)
        plot_filename = f"{title.replace(' ', '_').lower()}.svg"
        plt.savefig(os.path.join(output_path, plot_filename), format="svg", bbox_inches="tight")

