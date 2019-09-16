##################################################
### get out of sample rmse for susedcars data using linear regression

## import 
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

## read in data
cd = pd.read_csv("http://www.rob-mcculloch.org/data/susedcars.csv") #cd for car data
n = cd.values.shape[0]
print("n: ", n)

## pull off (price,mileage,year) and divide price and mileage by 1000
cd = cd[['price','mileage','year']]
cd['price'] = cd['price']/1000
cd['mileage'] = cd['mileage']/1000
print("cd dimensions: ", cd.shape)
print(cd.head()) # head just prints out the first few rows

## get numpy X and y
X = cd[['mileage','year']].values  #mileage and year columns as a numpy array
y = cd['price'].values #price as a numpy vector
#print("X shape: ", X.shape)
#print("y shape: ", y.shape)

## train/test split
nsamp = 500 # number of train/test plots
trainfrac = .75 # percent of data in train, rest is test
ntrain = math.floor(trainfrac*n)
ntest = n - ntrain
print("nsamp, n, ntrain, ntest: ", nsamp, n, ntrain, ntest)

## helper functions
def rmsef(y,yhat):
   'compute root mean squared error'
   return np.sqrt(np.mean((y-yhat)**2))

def trteind(n,ntrain):
   'gets indices for a train test split'
   itr = list(np.random.choice(n,ntrain,replace=False))
   alli = list(range(n))
   ite = list(set(alli) - set(itr))
   return {'tr':itr,'te':ite}

## store results
resv = np.zeros(nsamp)
resv_log = np.zeros(nsamp)
ysM = np.zeros((ntest,nsamp))
yhatM = np.zeros((ntest,nsamp))
yhatM_log = np.zeros((ntest,nsamp))

## loop over train/test
np.random.seed(99)
for i in np.arange(nsamp):
   if(not(i%5)): print("on sample: ",i)

   #train/test
   ii = trteind(n,ntrain)
   #print(ii)
   Xtr = X[ii['tr'],:]; ytr = y[ii['tr']]
   #print("Xtr shape: ", Xtr.shape, " ytr shape: ", ytr.shape)
   Xte = X[ii['te'],:]; yte = y[ii['te']]
   #print("Xte shape: ", Xte.shape, " yte shape: ", yte.shape)

   #convert y to logy
   ytr_log = np.log(ytr)

   #fit on logy
   lmmod1 = LinearRegression(fit_intercept=True)
   lmmod1.fit(Xtr, ytr_log)

   #fit on y
   lmmod2 = LinearRegression(fit_intercept=True)
   lmmod2.fit(Xtr, ytr)

   #predict on test
   yhatte = lmmod2.predict(Xte)
   yhatte_exp = np.exp(lmmod1.predict(Xte))

   #record loss
   resv[i] = rmsef(yte, yhatte)
   resv_log[i] = rmsef(yte, yhatte_exp)

   # keep sampled y and yhat
   ysM[:,i] = yte
   yhatM[:,i] = yhatte
   yhatM_log[:,i] = yhatte_exp
   

##plot results
plt.scatter(range(nsamp),resv)
plt.scatter(range(nsamp),resv_log)
plt.xlabel("# of samples")
plt.ylabel("rmse")
plt.title("rmse vs. # of samples")
plt.show()
plt.xlabel("y")
plt.ylabel("yhat")
plt.scatter(ysM.flatten(),yhatM.flatten())
plt.title("Regression using y")
plt.show()
plt.xlabel("y")
plt.ylabel("yhat")
plt.scatter(ysM.flatten(),yhatM_log.flatten())
plt.title("Regression using log y")
plt.show()

## write resv to file
resvDf = pd.DataFrame({'resv':resv, 'resv_log':resv_log})
resvDf.to_csv('resv-P.csv',index=False)

