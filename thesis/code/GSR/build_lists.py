import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


"""
Build the lists with mean and peak amplitude values for the training and testing sample of the SVM model.
Also plot the points in the end.
"""

#store the mean values for relaxed and stressed points in seperate lists
mean_values_relaxed = []
mean_values_stressed = []
mean_values_lol = []
mean_values_geo = []


# max_value - mean_value for stressed and relaxed
avg_dif_stressed = []
avg_dif_relaxed = []
avg_dif_lol = []
avg_dif_geo = []


# store (max_value - start value) for relaxed and stressed points in seperate lists
peak_ampl_relaxed = []
peak_ampl_stressed = []
peak_ampl_lol = []
peak_ampl_geo = []


# relaxed values folder paths
paths_relaxed = ['/home/gb/thesis/code/GSR/relaxed/GSRdata_s1p1v3.ods', '/home/gb/thesis/code/GSR/relaxed/GSRdata_s1p10v3.ods',
         '/home/gb/thesis/code/GSR/relaxed/GSRdata_s2p3v3.ods', '/home/gb/thesis/code/GSR/relaxed/GSRdata_s2p12v3.ods',
         '/home/gb/thesis/code/GSR/relaxed/GSRdata_S2p9v3.ods']

# stressed values folder paths
paths_stressed = ['/home/gb/thesis/code/GSR/stressed/GSRdata_s2p2v6.ods', '/home/gb/thesis/code/GSR/stressed/GSRdata_s2p5v7.ods'
                  , '/home/gb/thesis/code/GSR/stressed/GSRdata_s2p8v7.ods', '/home/gb/thesis/code/GSR/stressed/GSRdata_s3p6v6.ods',
                   '/home/gb/thesis/code/GSR/stressed/GSRdata_s3p7v6.ods', 
                   '/home/gb/thesis/code/GSR/stressed/GSRdata_s3p7v7.ods']

#lol players folder paths
paths_lol = ['/home/gb/thesis/code/GSR/lol/m0p0.ods', '/home/gb/thesis/code/GSR/lol/m15p1.ods',
             '/home/gb/thesis/code/GSR/lol/m16p2.ods', '/home/gb/thesis/code/GSR/lol/m16p4.ods', 
             '/home/gb/thesis/code/GSR/lol/m13p3.ods', '/home/gb/thesis/code/GSR/lol/m13p4.ods',
             '/home/gb/thesis/code/GSR/lol/m21p0.ods', '/home/gb/thesis/code/GSR/lol/m21p1.ods']

paths_geo = ['/home/gb/thesis/code/GSR/stressed/geo_s1.ods', '/home/gb/thesis/code/GSR/stressed/geo_s2.ods', 
             '/home/gb/thesis/code/GSR/relaxed/geo_r1.ods']



# # Read and calculate the relaxed values

# for i in range(len(paths_relaxed)):
#     excel_file_path = paths_relaxed[i]
#     df = pd.read_excel(excel_file_path)  # Read the entire Excel file
#     df_list = list(df['Y'])

#     mean_values_relaxed.append(statistics.mean(df_list))
#     avg_dif_relaxed.append(max(df_list) - statistics.mean(df_list))
#     peak_ampl_relaxed.append(max(df_list) - df_list[0])

# # Read and calculate the stressed values

# for i in paths_stressed:
#     excel_file_path = i
#     df = pd.read_excel(excel_file_path)  # Read the entire Excel file
#     df_list = list(df['Y'])

#     mean_values_stressed.append(statistics.mean(df_list))
#     avg_dif_stressed.append(max(df_list) - statistics.mean(df_list))
#     peak_ampl_stressed.append(max(df_list) - df_list[0])


# Read and calculate the lol values

# for i in paths_lol:
#     excel_file_path = i
#     df = pd.read_excel(excel_file_path)  # Read the entire Excel file
#     df_list = list(df['Y'])

#     mean_values_lol.append(statistics.mean(df_list))
#     avg_dif_lol.append(max(df_list) - statistics.mean(df_list))
#     peak_ampl_lol.append(max(df_list) - df_list[0])


# Read and calculate the geo values

for i in paths_geo:
    excel_file_path = i
    df = pd.read_excel(excel_file_path)  # Read the entire Excel file
    df_list = list(df['Y'])

    mean_values_geo.append(statistics.mean(df_list))
    avg_dif_geo.append(max(df_list) - statistics.mean(df_list))
    peak_ampl_geo.append(max(df_list) - df_list[0])


# print('mean values relaxed: ',mean_values_relaxed)
# print('avg_dif relaxed: ', avg_dif_relaxed)
# print('peak ampl relaxed: ', peak_ampl_relaxed)

# print('mean values stressed: ',mean_values_stressed)
# print('avg_dif stressed: ', avg_dif_stressed)
# print('peak ampl stressed: ', peak_ampl_stressed)

# print('mean values lol: ',mean_values_lol)
# print('avg_dif lol: ', avg_dif_lol)
# print('peak ampl lol: ', peak_ampl_lol)

print('mean values geo: ',mean_values_geo)
print('avg_dif geo: ', avg_dif_geo)
print('peak ampl geo: ', peak_ampl_geo)



# # plot the relaxed points 
# for i in range(len(mean_values_relaxed)) :
#     plt.scatter(mean_values_relaxed[i], avg_dif_relaxed[i], c='b')

# # plot the stressed points
# for i in range (len(mean_values_stressed)):
#         plt.scatter(mean_values_stressed[i], avg_dif_stressed[i], c='r')


# # plot the lol points
# for i in range (len(mean_values_lol)):
#         plt.scatter(mean_values_lol[i], avg_dif_lol[i], c='g')



plt.xlabel('mean (μS)')
plt.ylabel('avg diff (μS)')

plt.show()