from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

"""
SVM Model

Train with YAAD dataset
Test with lol and luna datasets

note : avg_dif = max_gsr_value - mean_value
"""

# relaxed 
mean_values_relaxed = [0.8793225230660457, 3.7665136449714143, 6.1791487952, 16.17389795664115, 0.4663673265843042]
avg_dif_relaxed = [0.25095590893992425, 2.099419392340636, 0.5239352047999999, 1.4493359523070524, 0.26039276979030385]
peak_ampl_relaxed = [0.43153558738528, 2.1608667102109305, 1.0682979999999995, 3.0123245281032016, 0.15166252592766405]

# relaxed data augmentation

mean_values_relaxed.append((mean_values_relaxed[0] + mean_values_relaxed[4]) / 2)
avg_dif_relaxed.append((avg_dif_relaxed[0] + avg_dif_relaxed[4]) / 2)
peak_ampl_relaxed.append ((avg_dif_relaxed[0] + avg_dif_relaxed[4]) / 2)

mean_values_relaxed.append((mean_values_relaxed[-1] + mean_values_relaxed[4]) / 2)
avg_dif_relaxed.append((avg_dif_relaxed[-1] + avg_dif_relaxed[4]) / 2)
peak_ampl_relaxed.append ((avg_dif_relaxed[-1] + avg_dif_relaxed[4]) / 2)

mean_values_relaxed.append(1.05)
avg_dif_relaxed.append(0.5678)
peak_ampl_relaxed.append(0.45)

mean_values_relaxed.append(1.21)
avg_dif_relaxed.append(0.391)
peak_ampl_relaxed.append(0)


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

# geo data
mean_values_geo = [5.113145087412279, 4.948352954504322, 3.514428015867776]
avg_dif_geo = [1.4978942650375613, 1.6991738528287783, 0.25058985568762404]
peak_ampl_geo = [2.78013556236238, 2.51894925018411, 0.6883578715554002]

# geo data augmentation



# SVM begins 

#list of lists for SVM
X = []
y = []
testing_data = []

# make the relaxed SVM training data
for i in range(len(mean_values_relaxed)) :
    X.append([mean_values_relaxed[i], avg_dif_relaxed[i]])
    y.append('relaxed')

# append the relaxed geo training values
X.append([mean_values_geo[2], avg_dif_geo[2]])
y.append('relaxed')


# make the stressed SVM training data
for i in range(len(mean_values_stressed)) :
    X.append([mean_values_stressed[i], avg_dif_stressed[i]])
    y.append('stressed')

# append the stressed geo training values
X.append([mean_values_geo[0], avg_dif_geo[0]])
y.append('stressed')

X.append([mean_values_geo[1], avg_dif_geo[1]])
y.append('stressed')



# make the SVM testing data
for i in range(len(mean_values_lol)) :
    testing_data.append([mean_values_lol[i], avg_dif_lol[i]])

for i in range(len(mean_values_luna)) :
    testing_data.append([mean_values_luna[i], avg_dif_luna[i]])

# Scale features (preprocessing)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
testing_data_scaled = scaler.fit_transform(testing_data)

# Initialize SVM model
svm_model = SVC(kernel="linear", C=1.0)
svm_model.fit(X_scaled, y)

# Predictions
y_pred = svm_model.predict(testing_data_scaled)
print("y_pred: ", y_pred)

# print stressed and relaxed testing percentage
stressed_counter = 0

for i in y_pred:
     if i == 'stressed':
          stressed_counter +=1
print ('Stressed percentage: ','{0:.2f}'.format(100 * (stressed_counter/len(y_pred))), '%')
print ('Relaxed percentage: ','{0:.2f}'.format(100 - (100 * (stressed_counter/len(y_pred)))), '%')


# accuracy score
y_test = ['stressed', 'stressed', 'stressed', 'relaxed', 'relaxed', 'relaxed', 'stressed', 'stressed', 'stressed', 'stressed']
print ('Accuracy score is : ', 100 * accuracy_score(y_test, y_pred[8:None]), '%')



# subplot for all the points

plt.subplot(1, 2, 1)

plt.title("Sample Points")

# plot the relaxed points 
for i in range(len(mean_values_relaxed)) :
    plt.scatter(mean_values_relaxed[i], avg_dif_relaxed[i], c='b')

# plot the stressed points
for i in range (len(mean_values_stressed)):
        plt.scatter(mean_values_stressed[i], avg_dif_stressed[i], c='r')


# plot the lol points
for i in range (len(mean_values_lol)):
        plt.scatter(mean_values_lol[i], avg_dif_lol[i], c='g')

# plot the luna points 
for i in range (len(mean_values_luna)):
        plt.scatter(mean_values_luna[i], avg_dif_luna[i], c='orange')    

# plot the geo points
for i in range (len(mean_values_geo)):
        plt.scatter(mean_values_geo[i], avg_dif_geo[i], c='magenta')    
    

# blank entries for legend

plt.scatter([], [], c='b', label='relaxed')
plt.scatter([], [], c='r', label='stressed')
plt.scatter([], [], c='g', label='lol players')
plt.scatter([], [], c='orange', label='covid testing')


plt.xlabel('mean (μS)')
plt.ylabel('avg diff (μS)')
plt.legend()


plt.subplot(1, 2, 2)

plt.title("Training and testing points")

# plot the training set

for i in range(len (X_scaled)):
     if y[i] == 'stressed':
        plt.scatter(X_scaled[i][0], X_scaled[i][1], c = 'red', edgecolors='k', s=50)
     else:
        plt.scatter(X_scaled[i][0], X_scaled[i][1], c = 'blue', edgecolors='k', s=50) 

# plot the testing set
        
for i in range(len(testing_data)):
     if y_pred[i] == 'stressed':
        plt.scatter(testing_data_scaled[i][0], testing_data_scaled[i][1], c='red', s=50, marker='x')
     else:
        plt.scatter(testing_data_scaled[i][0], testing_data_scaled[i][1], c='blue', s=50, marker='x')


# blank entries for legend

plt.scatter([], [], c='r', label='stressed training point')
plt.scatter([], [], c='b', label='relaxed training point')
plt.scatter([], [], c='r', label='stressed test point', marker='x')
plt.scatter([], [], c='b', label='relaxed test point', marker='x')


# plot the decision boundary only in linear kernel
W=svm_model.coef_[0]
I=svm_model.intercept_

a = -W[0]/W[1]
b = I[0]/W[1]
x = np.linspace(-0.04, 1.04, 50)

z = a*x - b

plt.plot(x, z)

plt.fill_between(x, z, -0.04, color='b', alpha=0.3)
plt.fill_between(x, z, 1.02, color='r', alpha=0.3)

print ('For regularization parameter C =', svm_model.get_params()['C'])
print('Decision boundary is the line : y = ', '{0:.2f}'.format (a), '* x + ', '{0:.2f}'.format (abs(b)))
# comment out until here when not in linear kernel 


plt.xlabel('mean (μS)')
plt.ylabel('avg diff (μS)')
plt.legend()

plt.figtext(0, 0.05, 'Accuracy score is : ' + str (100 * accuracy_score(y_test, y_pred[8:None])) + '%')
plt.figtext(0, 0.03, 'Stressed percentage: ' + str('{0:.2f}'.format(100 * (stressed_counter/len(y_pred)))) + '%')
plt.figtext(0, 0.01, 'Relaxed percentage: ' + str('{0:.2f}'.format(100 - (100 * (stressed_counter/len(y_pred))))) + '%')
plt.figtext(0.17, 0.01, 'Decision boundary is the line : y = ' + str('{0:.2f}'.format (a)) + '* x + ' + str('{0:.2f}'.format (abs(b))))

plt.tight_layout()
plt.show()