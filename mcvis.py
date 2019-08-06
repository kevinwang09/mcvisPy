import numpy as np
import pandas as pd
import statsmodels.api as sm



def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mcvis(x):
    n = x.shape[0]
    p = x.shape[1]
    
#    print("n is ", n)
#    print("p is ", p)
#    print("x is \n", np.round(x, 2))
#    print("Modified x's first column is \n", np.round(x[:, 0], 2))
#    print("Correlation matrix of x", np.corrcoef(x, rowvar = False))
    
    nexp = 1000
    v2_mat = np.ones((p, nexp))
    vif_mat = np.ones((p, nexp))
    
    for i in range(nexp):
        index = np.random.choice(range(n),n)
        x_sample = x[index, :] ## X1 in R
        x_sample_mean = x_sample.mean(axis = 0, keepdims=True)
        x_sample_centred = x_sample - x_sample_mean ## X2 in R
#        print(x_sample)
#        print("centred x is", np.round(x_sample_centred, 2))
        s = np.sqrt(np.sum(x_sample_centred ** 2, axis = 0)) ## s in R
#        print("s is", s)
        Z = x_sample_centred/s ## Z in R
#        print("Z is ", Z)
        x_norm = np.sqrt(np.sum(x_sample ** 2, axis = 0)) ## x_norm in R
#        print("x_norm is ", x_norm) 
        v = x_norm/s ## v in R
#        print("v is ", v)
        D = np.diag(v) ## D in R
#        print("D is ", D)
        Z1 = np.matmul(Z, D) ## Z1 in R
#        print("Z1 is ", Z1) 
        crossprodZ1 = np.matmul(np.transpose(Z1), Z1)
#        print("crossprodZ1 is ", crossprodZ1)
        eigens = np.linalg.svd(crossprodZ1, compute_uv = False) ## Not in R
#        print("eigens is", eigens)
        v2 = 1/eigens ## Not in R
#        print("v2 is ", v2)
        vif = np.diag(np.linalg.inv(crossprodZ1)) ## Not in R
#        print("vif is ", vif)
        
        v2_mat[:,i] = v2 ## v2 in R
        vif_mat[:,i] = vif ## vif in R
            
    
#    tor = np.ones((p, 1))
    
    index_list = list(chunks(range(1000), 100)) ## indexList in R
    tstat_mat = np.ones((p, 10))
    tor = np.ones((p, p))
    
    for j in range(p):
        for this_index in range(10):
            lmy = v2_mat[j,index_list[this_index]]
            lmx = sm.add_constant(np.transpose(vif_mat[:,index_list[this_index]]))
            lm_obj = sm.OLS(lmy,lmx)
            lm_result = lm_obj.fit()
            lm_tvalues = lm_result.tvalues
            print("tvalues", np.round(lm_tvalues, 2))
            tstat_mat[:, this_index] = lm_tvalues[1:]
        print("tstat_mat", np.round(tstat_mat, 2))
        tor[j, :] = np.mean(tstat_mat ** 2, axis = 1)


    tor_rowsums = np.sum(tor, axis = 1, keepdims = True) ## not in R
    mc = tor/tor_rowsums
    mc_rownames = range(p+1)[1:]
    
    mc_pd = pd.DataFrame(mc,
                         columns = mc_rownames,
                         index = mc_rownames[::-1])
       
    result = dict();
    result["mc_pd"] = mc_pd
    result["tor"] = tor
    
    return result
    

n = 20
p = 5

x = np.random.normal(0, 1, n*p).reshape(n, p)
# y = np.random.normal(0, 1, n*p)

x[:, 0] = x[:, 1] + np.random.normal(0, 0.5, n)


res = mcvis(x)
