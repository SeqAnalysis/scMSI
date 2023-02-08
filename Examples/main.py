import numpy as np
from Codes.scMSI_Mixture import SCMSIMixture
from multiprocessing import Pool
from Codes.scMSI_Mixture import delete


if __name__ == '__main__':

    parallelized = False
    num_process = 3
    t = np.loadtxt('t.txt', dtype=int)
    pt=np.loadtxt('point.txt')
    thr=["thr1","thr2","thr3"]
    X = np.loadtxt('tumor_0.5.txt', dtype=int)
    X = X.reshape(-1, 1)
    c = np.loadtxt('tum_0.5.txt')
    if not parallelized:
        dpgm = SCMSIMixture(n_components=3, cov_type='full', max_iter=100,a0=0.1,
                                        fraction_type='dirichlet_distribution').fit(X, c,thr[0])
        print("means:")
        print(dpgm.means_)
        print("Mixed proportion:")
        print(dpgm.fractions_)
        print("covs:")
        for i in range(len(dpgm.covs_)):
            print(np.sqrt(dpgm.covs_[i]))
    else:
        dpgmm = SCMSIMixture(n_components=3, cov_type='full', max_iter=100,a0=0.1,
                                        fraction_type='dirichlet_distribution')
        p = Pool(processes=num_process)
        p.starmap(dpgmm.fit, [((delete(t[i])).reshape(-1, 1), delete(pt[i]), thr[i]) for i in range(t.shape[0])])
        p.close()

