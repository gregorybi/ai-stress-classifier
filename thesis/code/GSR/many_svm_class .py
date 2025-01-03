from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
import tkinter as tk


"""
SVM Model

Train with YAAD dataset
Test with lol and luna datasets

Display 6 graphs for 6 SVM configs

note 1 : avg_dif = max_gsr_value - mean_value
note 2 : do not use peak ampl for classification
"""

# relaxed 
mean_values_relaxed = [0.8793225230660457, 3.7665136449714143, 6.1791487952, 16.17389795664115, 0.4663673265843042,
                       1.745, 1.77, 1.885, 1.6875, 1.78625, 1.64, 1.6, 1.68, 1.71]

avg_dif_relaxed = [0.25095590893992425, 2.099419392340636, 0.5239352047999999, 1.4493359523070524, 0.26039276979030385,
                   0.27, 0.17, 0.28, 0.215, 0.2475, 0.19, 0.16, 0.15, 0.14]

peak_ampl_relaxed = [0.43153558738528, 2.1608667102109305, 1.0682979999999995, 3.0123245281032016, 0.15166252592766405,
                     0.21, 0.195, 0.395, 0.11, 0.2525, 0.2, 0.15, 0.3, 0.22]

# relaxed data augmentation

mean_values_relaxed.append((mean_values_relaxed[0] + mean_values_relaxed[4]) / 2)
avg_dif_relaxed.append((avg_dif_relaxed[0] + avg_dif_relaxed[4]) / 2)
peak_ampl_relaxed.append ((avg_dif_relaxed[0] + avg_dif_relaxed[4]) / 2)

mean_values_relaxed.append((mean_values_relaxed[-1] + mean_values_relaxed[4]) / 2)
avg_dif_relaxed.append((avg_dif_relaxed[-1] + avg_dif_relaxed[4]) / 2)
peak_ampl_relaxed.append ((avg_dif_relaxed[-1] + avg_dif_relaxed[4]) / 2)

mean_values_relaxed.extend([1.05, 1.21, 4.116133816659943, 6.124382804276595, 3.0934694055161644, 3.4213327396653073, 5.939634580559892, 5.111708918949979, 2.993414621733558, 4.218144923078794])
avg_dif_relaxed.extend([0.5678, 0.391, 0.25292070900317026, 0.3769266773556206, 0.3130314970301805, 0.5234493480515577, 0.49702327860473255, 0.37504594174858275, 0.28882210326447244, 0.34514921561146317])
# peak_ampl_relaxed.extend([0.45, 0.391])

mean_values_relaxed.extend([4.4753176326715, 4.858884103936029, 4.973408708781862, 5.981965250310491, 2.102645210355365, 2.8689829589506104, 1.8257226248638905, 4.079245929406925, 4.397861323020496, 5.293063724615125])
avg_dif_relaxed.extend([0.35406910443452067, 0.42704167109792257, 0.3678203528663922, 0.24188832050792275, 0.2678826597100129, 0.2471387917702263, 0.47048258128368686, 0.4994105160425289, 0.2224640439680965, 0.2701112957619477])


# append relaxed geo data
mean_values_relaxed.append(3.514428015867776)
avg_dif_relaxed.append( 0.25058985568762404)
peak_ampl_relaxed.append(0.6883578715554002)


# stressed
mean_values_stressed = [0.7573444213667326, 3.428098384263275, 6.707323366874757, 20.35445386333723, 
                        2.146846659341597, 3.2974970959672487]
avg_dif_stressed = [0.14352026837998733, 1.5136898620397048, 5.182395046863842, 3.963039321747871, 
                    0.707501064381943, 0.37112222110017123]
peak_ampl_stressed = [0.021446654684622923, 1.4642630798135996, 7.867657194668659, 0.0, 0.0, 0.7807877204621199]


# stressed data augmentation
mean_values_stressed.append((mean_values_stressed[2] + mean_values_stressed[3]) / 2)
avg_dif_stressed.append((avg_dif_stressed[2] + avg_dif_stressed[3]) / 2)
peak_ampl_stressed.append ((avg_dif_stressed[2] + avg_dif_stressed[3]) / 2)


mean_values_stressed.append((mean_values_stressed[1] + mean_values_stressed[4]) / 2)
avg_dif_stressed.append((avg_dif_stressed[1] + avg_dif_stressed[4]) / 2)
peak_ampl_stressed.append ((avg_dif_stressed[1] + avg_dif_stressed[4]) / 2)


mean_values_stressed.append((mean_values_stressed[-1] + mean_values_stressed[4]) / 2)
avg_dif_stressed.append((avg_dif_stressed[-1] + avg_dif_stressed[4]) / 2)
peak_ampl_stressed.append ((avg_dif_stressed[-1] + avg_dif_stressed[4]) / 2)


mean_values_stressed.append((mean_values_stressed[5] + mean_values_stressed[1]) / 2)
avg_dif_stressed.append((avg_dif_stressed[5] + avg_dif_stressed[1]) / 2)
peak_ampl_stressed.append ((avg_dif_stressed[5] + avg_dif_stressed[1]) / 2)

# append stressed geo data
mean_values_stressed.append(5.113145087412279)
avg_dif_stressed.append(1.4978942650375613)
peak_ampl_stressed.append (2.78013556236238)

mean_values_stressed.append(4.948352954504322)
avg_dif_stressed.append(1.6991738528287783)
peak_ampl_stressed.append (2.51894925018411)


mean_values_stressed.extend([5.091780363050294, 4.703750746258344, 6.237042357367219, 3.3077669469582887, 5.144695629536651, 6.053568981409104, 4.109583716921161, 4.16502258058147, 4.6179198459452815, 6.340622723862694])
avg_dif_stressed.extend([4.857486153546329, 4.773921860828523, 2.871058388460217, 4.890305857470006, 3.315844283276637, 4.370486786791529, 4.773114318562827, 1.0431080566225446, 4.0876332854405355, 4.801709365591265])

mean_values_stressed.extend([6.748630258592576, 12.770563615257343, 13.726199578791089, 11.272537767060673, 19.69549225425153, 15.193501191759923, 16.66276487791547, 8.058955257559838, 14.912773203728449, 16.767975045830738])
avg_dif_stressed.extend([4.267674883375358, 4.099770753417324, 5.019781777783692, 5.082033088282, 5.123832388805193, 5.068598968073622, 4.361593944336308, 4.244387728116694, 4.588475087625794, 4.379669882905299])

# lol
mean_values_lol = [2.3479595873359393, 2.210763975361086, 5.634308730544468, 3.227823399189036, 2.0385365245208296,
                   3.860936888920899, 4.944780430155195, 2.0487880117323973]

avg_dif_lol = [4.4833962111541705, 1.7608814151386638, 1.0610543844914417, 0.9745552104745241, 0.36725812745747044,
               0.643478944080941, 1.609049214980275, 0.7610799843799927]

peak_ampl_lol = [0.1249424900670002, 1.1732319711270698, 0.8476660571871903, 0.889813770685, 0.52618229516653,
                 1.03122979618971, 1.81824520187436, 1.28330166799702]

# luna data
mean_values_luna = [1.56, 2.58, 2.28, 1.63, 1.86, 1.91, 2.28, 2.05, 1.87, 9.44]

avg_dif_luna = [0.06, 0.37, 0.71, 0.16, 0.38, 0.18, 0.71, 0.22, 1.39, 2.93]

peak_ampl_luna = [0.14, 0.88, 1.06, 0.01, 0.41, 0.38, 1.06, 0.45, 1.3, 6.07]

# SVM begins 

# list of lists for SVM
X = []
y = []
testing_data = []

# make the relaxed SVM training data
for i in range(len(mean_values_relaxed)) :
    X.append([mean_values_relaxed[i], avg_dif_relaxed[i]])
    y.append('b')         

# make the stressed SVM training data
for i in range(len(mean_values_stressed)) :
    X.append([mean_values_stressed[i], avg_dif_stressed[i]])
    y.append('r')         

# make the SVM testing data
for i in range(len(mean_values_lol)) :
    testing_data.append([mean_values_lol[i], avg_dif_lol[i]])

for i in range(len(mean_values_luna)) :
    testing_data.append([mean_values_luna[i], avg_dif_luna[i]])
    
# Scale features (preprocessing)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
testing_data_scaled = scaler.fit_transform(testing_data)

C = 1.0  # SVM regularization parameter
models1 = [
   SVC(kernel="linear", C=C, probability=True),
   LinearSVC(C=C, max_iter=10000, dual="auto"),
   SVC(kernel="rbf", gamma=2.0, C=C, probability=True),
   SVC(kernel="poly", degree=5, gamma="scale", C=C, probability=True),
   SVC(kernel="poly", degree=10, gamma="scale", C=C, probability=True),
   SVC(kernel="sigmoid", gamma="scale", C=C, probability=True),
   NuSVC(nu=0.2, probability=True),
   NuSVC(nu=0.01, kernel='sigmoid', probability=True),
   NuSVC(nu=0.23, kernel='poly', probability=True),
]
models = list((clf.fit(X_scaled, y) for clf in models1))

# Predictions
y_pred = list((clf.predict(testing_data_scaled) for clf in models))


# title for the plots
titles = [
    "SVM with linear kernel",
    "LinearSVM (linear kernel)",
    "SVM with RBF kernel and gamma = " + str(models1[2].get_params()['gamma']),
    "SVM with polynomial (degree " + str(models1[3].get_params()['degree']) + ") kernel",
    "SVM with polynomial (degree " + str(models1[4].get_params()['degree']) + ") kernel",
    'SVM with sigmoid kernel', 
    'NuSVM with '  + str(models1[6].get_params()['kernel']) + ' kernel, nu= ' + str(models1[6].get_params()['nu']) + ' ,gamma=' + str(models1[6].get_params()['gamma']),
    'NuSVM with '  + str(models1[7].get_params()['kernel']) + ' kernel and nu ' + str(models1[7].get_params()['nu']),
    'NuSVM with '  + str(models1[8].get_params()['kernel']) + ' kernel and nu ' + str(models1[8].get_params()['nu']),
]

# Set-up 3x2 grid for plotting.
fig, sub = plt.subplots(3, 3)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X3 = np.array(np.array(X_scaled))
X4 = np.array(np.array(testing_data_scaled))


X0, X1 =  X3[:, 0], X3[:, 1]
X5, X6 = X4[:, 0], X4[:, 1]

# statistics list with ercentages
percentages = []

# luna data classification list
y_test = ['r', 'r', 'r', 'b', 'b', 'b', 'r', 'r', 'r', 'r']

# numeric luna data classification list for f1 score 
y_test_ar = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]


# Calculate the AUC-ROC score and place it in a list for each case
auc_roc_list = []


for clf, title, ax, i in zip(models, titles, sub.flatten(), range(len(y_pred))):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X3,
        response_method="predict",        
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel='mean (μS)',
        ylabel='avg diff (μS)'
    )
    ax.scatter(X0, X1, c=y, s=20, edgecolors="k")

    # stressed counter for percentage calculation
    stressed_counter = 0
    
    for j in range(len(y_pred[i])):
        if y_pred[i][j] == 'r': 
            ax.scatter(X5[j], X6[j], s=20, color='r', marker='x')
            stressed_counter +=1
        else:
            ax.scatter(X5[j], X6[j], s=20, color='b', marker='x')

    percentages.append(100* stressed_counter / len(y_pred[i]))

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

# Set-up 2x4 grid for plotting.
fign, subn = plt.subplots(3, 3)
plt.subplots_adjust(wspace=1.0, hspace=1.0)

for clf, title, ax, i in zip(models1, titles, subn.flatten(), range(len(titles))):
    if title == "LinearSVM (linear kernel)":
        continue
    else:
        # Get predicted probabilities for the positive class
        y_pred_proba = clf.predict_proba(testing_data_scaled)[:, 1]

        auc_roc = roc_auc_score(y_test, y_pred_proba[8:None])
        auc_roc_list.append(auc_roc)

        # Compute ROC curve
        fpr, tpr, thres = roc_curve(y_test_ar, y_pred_proba[8:None])

        print("Threshold is: ", thres)
      
        ax.plot(fpr, tpr, color='b', lw=2, label=f"AUC = {auc_roc:.2f}")
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("(ROC) Curve for " + title, fontsize=7)
        ax.legend(loc="lower right")
        ax.grid(True)

plt.show()

# f1 score calculation 
f1_pred = []
f1_percentages = []


for i in range(len(titles)):
    for j in range(len(y_test)):
        if y_pred[i][8+j] == 'b':
            f1_pred.append(0)
        else:
            f1_pred.append(1)
    
    f1_percentages.append(100 * f1_score(y_test_ar, f1_pred))
    f1_pred.clear()


# statistics section

# display statistics with tkinter

root = tk.Tk()
 
# specify size of window.
root.geometry("700x700")
 
# Create text widget and specify size.
T = tk.Text(root, height = 35, width = 52)

 
# Create label
l = tk.Label(root, text = "SVM Statistics")
l.config(font =("Courier", 14))
 
Fact1a = 'SVM type: ' + titles[0] + '\n' + 'stressed percentage: ' + str('{0:.2f}'.format(percentages[0])) + '%\n' 
Fact1b = 'Relaxed percentage: ' + str('{0:.2f}'.format(100 - percentages[0])) + '%\n'
Fact1c = 'Accuracy score: ' + str(100 * accuracy_score(y_test, y_pred[0][8:None])) + '%\n'
Fact1d = 'F1 score: ' + str('{0:.2f}'.format(f1_percentages[0])) + '%\n'
Fact1e = "AUC-ROC Score: " + str('{0:.4f}'.format(auc_roc_list[0])) + '\n\n'

Fact2a = 'SVM type: ' + titles[1] + '\n' + 'stressed percentage: ' + str('{0:.2f}'.format(percentages[1])) + '%\n'
Fact2b = 'Relaxed percentage: ' + str('{0:.2f}'.format(100 - percentages[1])) + '%\n'
Fact2c = 'Accuracy score: ' + str(100 * accuracy_score(y_test, y_pred[1][8:None])) + '%\n'
Fact2d = 'F1 score: ' + str('{0:.2f}'.format(f1_percentages[1])) + '%\n'
Fact2e = 'AUC-ROC Score cannot be calculated in LinearSVM\n\n'

Fact3a = 'SVM type: ' + titles[2] + '\n' + 'stressed percentage: ' + str('{0:.2f}'.format(percentages[2])) + '%\n' 
Fact3b = 'Relaxed percentage: ' + str('{0:.2f}'.format(100 - percentages[2])) + '%\n'
Fact3c = 'Accuracy score: ' + str(100 * accuracy_score(y_test, y_pred[2][8:None])) + '%\n'
Fact3d = 'F1 score: ' + str('{0:.2f}'.format(f1_percentages[2])) + '%\n'
Fact3e = "AUC-ROC Score: " + str('{0:.4f}'.format(auc_roc_list[1])) + '\n\n'

Fact4a = 'SVM type: ' + titles[3] + '\n' + 'stressed percentage: ' + str('{0:.2f}'.format(percentages[3])) + '%\n'
Fact4b = 'Relaxed percentage: ' + str('{0:.2f}'.format(100 - percentages[3])) + '%\n'
Fact4c = 'Accuracy score: ' + str(100 * accuracy_score(y_test, y_pred[3][8:None])) + '%\n'
Fact4d = 'F1 score: ' + str('{0:.2f}'.format(f1_percentages[3])) + '%\n'
Fact4e = "AUC-ROC Score: " + str('{0:.4f}'.format(auc_roc_list[2])) + '\n\n'

Fact5a = 'SVM type: ' + titles[4] + '\n' + 'stressed percentage: ' + str('{0:.2f}'.format(percentages[4])) + '%\n'
Fact5b = 'Relaxed percentage: ' + str('{0:.2f}'.format(100 - percentages[4])) + '%\n'
Fact5c = 'Accuracy score: ' + str(100 * accuracy_score(y_test, y_pred[4][8:None])) + '%\n'
Fact5d = 'F1 score: ' + str('{0:.2f}'.format(f1_percentages[4])) + '%\n'
Fact5e = "AUC-ROC Score: " + str('{0:.4f}'.format(auc_roc_list[3])) + '\n\n'

Fact6a = 'SVM type: ' + titles[5] + '\n' + 'stressed percentage: ' + str('{0:.2f}'.format(percentages[5])) + '%\n'
Fact6b = 'Relaxed percentage: ' + str('{0:.2f}'.format(100 - percentages[5])) + '%\n'
Fact6c = 'Accuracy score: ' + str(100 * accuracy_score(y_test, y_pred[5][8:None])) + '%\n'
Fact6d = 'F1 score: ' + str('{0:.2f}'.format(f1_percentages[5])) + '%\n'
Fact6e = "AUC-ROC Score: " + str('{0:.4f}'.format(auc_roc_list[4])) + '\n\n'

Fact7a = 'SVM type: ' + titles[6] + '\n' + 'stressed percentage: ' + str('{0:.2f}'.format(percentages[6])) + '%\n'
Fact7b = 'Relaxed percentage: ' + str('{0:.2f}'.format(100 - percentages[6])) + '%\n'
Fact7c = 'Accuracy score: ' + str(100 * accuracy_score(y_test, y_pred[6][8:None])) + '%\n'
Fact7d = 'F1 score: ' + str('{0:.2f}'.format(f1_percentages[6])) + '%\n'
Fact7e = "AUC-ROC Score: " + str('{0:.4f}'.format(auc_roc_list[5])) + '\n\n'

Fact8a = 'SVM type: ' + titles[7] + '\n' + 'stressed percentage: ' + str('{0:.2f}'.format(percentages[7])) + '%\n'
Fact8b = 'Relaxed percentage: ' + str('{0:.2f}'.format(100 - percentages[7])) + '%\n'
Fact8c = 'Accuracy score: ' + str(100 * accuracy_score(y_test, y_pred[7][8:None])) + '%\n'
Fact8d = 'F1 score: ' + str('{0:.2f}'.format(f1_percentages[7])) + '%\n'
Fact8e = "AUC-ROC Score: " + str('{0:.4f}'.format(auc_roc_list[6])) + '\n\n'

Fact9a = 'SVM type: ' + titles[8] + '\n' + 'stressed percentage: ' + str('{0:.2f}'.format(percentages[8])) + '%\n'
Fact9b = 'Relaxed percentage: ' + str('{0:.2f}'.format(100 - percentages[8])) + '%\n'
Fact9c = 'Accuracy score: ' + str(100 * accuracy_score(y_test, y_pred[8][8:None])) + '%\n'
Fact9d = 'F1 score: ' + str('{0:.2f}'.format(f1_percentages[8])) + '%\n'
Fact9e = "AUC-ROC Score: " + str('{0:.4f}'.format(auc_roc_list[7])) + '\n\n'

 
# Create an Exit button.
b2 = tk.Button(root, text = "Exit", command = root.destroy) 

 
l.pack()
T.pack()
b2.pack()
 
# Insert The Fact.
T.insert(tk.END, 'Total relaxed training samples: ' + str(len(mean_values_relaxed)) + '\n')
T.insert(tk.END, 'Total stressed training samples: ' + str(len(mean_values_stressed)) + '\n\n')
T.insert(tk.END, Fact1a + Fact1b + Fact1c + Fact1d + Fact1e)
T.insert(tk.END, Fact2a + Fact2b + Fact2c + Fact2d + Fact2e)
T.insert(tk.END, Fact3a + Fact3b + Fact3c + Fact3d + Fact3e)
T.insert(tk.END, Fact4a + Fact4b + Fact4c + Fact4d + Fact4e)
T.insert(tk.END, Fact5a + Fact5b + Fact5c + Fact5d + Fact5e)
T.insert(tk.END, Fact6a + Fact6b + Fact6c + Fact6d + Fact6e)
T.insert(tk.END, Fact7a + Fact7b + Fact7c + Fact7d + Fact7e)
T.insert(tk.END, Fact8a + Fact8b + Fact8c + Fact8d + Fact8e)
T.insert(tk.END, Fact9a + Fact9b + Fact9c + Fact9d + Fact9e)

T.config(state='disabled')
 
tk.mainloop()