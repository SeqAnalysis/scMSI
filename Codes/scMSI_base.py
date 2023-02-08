import warnings
from abc import ABCMeta, abstractmethod
from time import time
import numpy as np
from scipy.special import logsumexp
from sklearn import cluster
from sklearn.base import BaseEstimator
from sklearn.base import DensityMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

def check_param_shape(param, param_shape, name):

    param = np.array(param)
    if param.shape != param_shape:
        raise ValueError(
            "The parameter '%s' should have the shape of %s, but got %s"
            % (name, param_shape, param.shape)
        )


class SCMSIBase(DensityMixin, BaseEstimator, metaclass=ABCMeta):

    def __init__(
            self,
            n_components,
            tol,
            reg_covar,
            max_iter,
            n_init,
            init_params,
            random_state,
            warm_start,
            diffuse,
            diffuse_val,
    ):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.diffuse = diffuse
        self.diffuse_val = diffuse_val

    def check_init_parameters(self, X):

        if self.n_components < 1:
            raise ValueError(
                "Invalid value for 'n_components': %d "
                % self.n_components
            )

        if self.tol < 0.0:
            raise ValueError(
                "Invalid value for 'tol': %.5f "
                % self.tol
            )

        if self.n_init < 1:
            raise ValueError(
                "Invalid value for 'n_init': %d"
                % self.n_init
            )

        if self.max_iter < 1:
            raise ValueError(
                "Invalid value for 'max_iter': %d "
                % self.max_iter
            )

        if self.reg_covar < 0.0:
            raise ValueError(
                "Invalid value for 'reg_covar': %.5f "
                % self.reg_covar
            )

        self.check_parameters(X)

    @abstractmethod
    def check_parameters(self, X):

        pass

    def init_parameters(self, X, random_state):

        n_samples, _ = X.shape
        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components))
            label = (
                cluster.KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                    .fit(X)
                    .labels_
            )
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError(
                "Unimplemented initialization method '%s'" % self.init_params
            )

        self.init(X, resp)

    @abstractmethod
    def init(self, X, resp):

        pass

    def fit(self, X,c,threadName,y=None):

        self.fit_predict(X,c,threadName,y)
        return self

    def fit_predict(self, X,c,threadName,y=None):

        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self.check_init_parameters(X)

        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False
        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self.init_beg(init)

            if do_init:
                self.init_parameters(X, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound
                log_prob_norm, log_resp = self._e_step(X)
                label_T=log_resp.argmax(axis=1)
                np.savetxt('D:\MSI\data_msisensor\laaa.txt', label_T, fmt='%d')
                self._m_step(X,c, log_resp)
                lower_bound = self.compute_lower_bound(log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self.iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    print("lower_bound：")
                    print(lower_bound)
                    print(threadName)
                    break

            self.init_end(lower_bound)

            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_params = self.get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn(
                "did not converge, try different init parameters"
                % (init + 1),ConvergenceWarning,)

        self.set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound


        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)

    def _e_step(self, X):
        log_prob_norm, log_resp = self.compute_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    @abstractmethod
    def _m_step(self, X,c, log_resp):

        pass

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def set_parameters(self, params):
        pass


    def predict(self, X):

        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        #_, log_resp = self.compute_log_prob_resp(X)
        #return log_resp.argmax(axis=1)
        #c=np.exp(log_resp)
        #return c
        return self.compute_fractional_log_prob(X).argmax(axis=1)



    def compute_fractional_log_prob(self, X):

        return self.et_log_prob(X) + self.et_log_fractions()

    @abstractmethod
    def et_log_fractions(self):

        pass

    @abstractmethod
    def et_log_prob(self, X):

        pass

    def compute_log_prob_resp(self, X):
        fractional_log_prob = self.compute_fractional_log_prob(X)
        log_prob_norm = logsumexp(fractional_log_prob, axis=1)
        with np.errstate(under="ignore"):
            log_resp = fractional_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def init_beg(self, n_init):
        if self.diffuse == 1:
            print("Initialization %d" % n_init)
        elif self.diffuse >= 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time

    def iter_end(self, n_iter, diff_ll):
        if n_iter % self.diffuse_val == 0:
            if self.diffuse == 1:
                print("  Iteration %d" % n_iter)
            elif self.diffuse >= 2:
                cur_time = time()
                print(
                    "  Iteration %d\t time lapse %.5fs\t ll change %.5f"
                    % (n_iter, cur_time - self._iter_prev_time, diff_ll)
                )
                self._iter_prev_time = cur_time

    def init_end(self, ll):
        if self.diffuse == 1:
            print("Initialization converged: %s" % self.converged_)
        elif self.diffuse >= 2:
            print(
                "Initialization converged: %s\t time lapse %.5fs\t ll %.5f"
                % (self.converged_, time() - self._init_prev_time, ll)
            )

