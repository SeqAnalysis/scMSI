import numpy as np
from scipy.special import betaln, digamma, gammaln
from scipy import linalg
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_array
from Codes.scMSI_base import SCMSIBase,check_param_shape
from gurobipy import*

def compute_covs_f(resp, X, nk, means, reg_covar):
    n_components, n_features = means.shape
    covs = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covs[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covs[k].flat[:: n_features + 1] += reg_covar
    return covs


def compute_covs_t(resp, X, nk, means, reg_covar):
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    cov = avg_X2 - avg_means2
    cov /= nk.sum()
    cov.flat[:: len(cov) + 1] += reg_covar
    return cov


def compute_covs_d(resp, X, nk, means, reg_covar):
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar


def compute_covs_s(resp, X, nk, means, reg_covar):
    return compute_covs_d(resp, X, nk, means, reg_covar).mean(1)


def compute_parameters(X, resp, reg_covar, cov_type):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]

    covs = {
        "full": compute_covs_f,
        "tied": compute_covs_t,
        "diag": compute_covs_d,
        "spherical": compute_covs_s,
    }[cov_type](resp, X, nk, means, reg_covar)
    return nk, means, covs

def delete(X):
    a = []
    for i in range(len(X)):
        if (X[i] != 0):
            a.append(X[i])
    a = np.array(a)
    return (a)

def compute_precision(covs, cov_type):
    compute_precision_error = (
        "Try to decrease the number of components, or increase reg_covar."
    )

    if cov_type == "full":
        n_components, n_features, _ = covs.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, cov in enumerate(covs):
            try:
                cov_chol = linalg.cholesky(cov, lower=True)
            except linalg.LinAlgError:
                raise ValueError(compute_precision_error)
            precisions_chol[k] = linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T
    elif cov_type == "tied":
        _, n_features = covs.shape
        try:
            cov_chol = linalg.cholesky(covs, lower=True)
        except linalg.LinAlgError:
            raise ValueError(compute_precision_error)
        precisions_chol = linalg.solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T
    else:
        if np.any(np.less_equal(covs, 0.0)):
            raise ValueError(compute_precision_error)
        precisions_chol = 1.0 / np.sqrt(covs)
    return precisions_chol



def compute_log_det(matrix_chol, cov_type, n_features):
    if cov_type == "full":
        n_components, _, _ = matrix_chol.shape
        log_det_chol = np.sum(
            np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
        )

    elif cov_type == "tied":
        log_det_chol = np.sum(np.log(np.diag(matrix_chol)))

    elif cov_type == "diag":
        log_det_chol = np.sum(np.log(matrix_chol), axis=1)

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol


def compute_log_prob(X, means, precisions_chol, cov_type):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    log_det = compute_log_det(precisions_chol, cov_type, n_features)

    if cov_type == "full":
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif cov_type == "tied":
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif cov_type == "diag":
        precisions = precisions_chol ** 2
        log_prob = (
            np.sum((means ** 2 * precisions), 1)
            - 2.0 * np.dot(X, (means * precisions).T)
            + np.dot(X ** 2, precisions.T)
        )

    elif cov_type == "spherical":
        precisions = precisions_chol ** 2
        log_prob = (
            np.sum(means ** 2, 1) * precisions
            - 2 * np.dot(X, means.T * precisions)
            + np.outer(row_norms(X, squared=True), precisions)
        )
    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det



def check_precision(precision, cov_type):

    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be positive" % cov_type)


def precision_mat(precision, cov_type):

    if not (
        np.allclose(precision, precision.T) and np.all(linalg.eigvalsh(precision) > 0.0)
    ):
        raise ValueError(
            "'%s precision' should be symmetric, positive-definite" % cov_type
        )


def compute_log_dirichlet(dirichlet_concentration):
    return gammaln(np.sum(dirichlet_concentration)) - np.sum(
        gammaln(dirichlet_concentration)
    )


def compute_log_wishart(degrees_of_freedom, log_det_precisions_chol, n_features):

    return -(
            degrees_of_freedom * log_det_precisions_chol
            + degrees_of_freedom * n_features * 0.5 * math.log(2.0)
            + np.sum(gammaln(0.5 * (degrees_of_freedom - np.arange(n_features)[:, np.newaxis])),0,))


class SCMSIMixture(SCMSIBase):
    def __init__(
            self,
            *,
            n_components=1,
            cov_type="full",
            tol=0.0001,
            reg_covar=1e-6,
            max_iter=100,
            n_init=1,
            init_params="kmeans",
            fraction_type="dirichlet_process",
            a0=None,
            b0=None,
            m0=None,
            v0=None,
            w0=None,
            random_state=42,
            warm_start=False,
            diffuse=0,
            diffuse_val=10,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            diffuse=diffuse,
            diffuse_val=diffuse_val,
        )

        self.cov_type = cov_type
        self.fraction_type = fraction_type#权重分布类型
        self.a0 = a0
        self.b0 = b0
        self.m0 = m0
        self.v0 = v0
        self.w0 = w0

    def check_parameters(self, X):

        if self.cov_type not in ["spherical", "tied", "diag", "full"]:
            raise ValueError(
                "Invalid value for 'cov_type': %s "
                % self.cov_type
            )

        if self.fraction_type not in [
            "dirichlet_process",
            "dirichlet_distribution",
        ]:
            raise ValueError(
                "Invalid value for 'fraction_type': %s "
                % self.fraction_type
            )

        self.check_fractions()
        self.check_means(X)
        self.check_precision(X)
        self.check_cov(X)

    def check_fractions(self):

        if self.a0 is None:
            self.fraction_concentration_prior_ = 1.0 / self.n_components
        elif self.a0 > 0.0:
            self.fraction_concentration_prior_ = self.a0
        else:
            raise ValueError(
                "a0 should be greater than 0."
                % self.a0
            )

    def check_means(self, X):
        _, n_features = X.shape

        if self.b0 is None:
            self.mean_precision_prior_ = 1.0
        elif self.b0 > 0.0:
            self.mean_precision_prior_ = self.b0
        else:
            raise ValueError(
                "b0 greater than 0."
                % self.b0
            )

        if self.m0 is None:
            self.mean_prior_ = X.mean(axis=0)
        else:
            self.mean_prior_ = check_array(
                self.m0, dtype=[np.float64, np.float32], ensure_2d=False
            )
            check_param_shape(self.mean_prior_, (n_features,), "means")

    def check_precision(self, X):

        _, n_features = X.shape

        if self.v0 is None:
            self.degrees_of_freedom_prior_ = n_features
        elif self.v0 > n_features - 1.0:
            self.degrees_of_freedom_prior_ = self.v0
        else:
            raise ValueError(
                "v0 should be greater than %d."
                % (n_features - 1, self.v0)
            )

    def check_cov(self, X):

        _, n_features = X.shape

        if self.w0 is None:
            self.cov_prior_ = {
                "full": np.atleast_2d(np.cov(X.T)),
                "tied": np.atleast_2d(np.cov(X.T)),
                "diag": np.var(X, axis=0, ddof=1),
                "spherical": np.var(X, axis=0, ddof=1).mean(),
            }[self.cov_type]

        elif self.cov_type in ["full", "tied"]:
            self.cov_prior_ = check_array(
                self.w0, dtype=[np.float64, np.float32], ensure_2d=False
            )
            check_param_shape(
                self.cov_prior_,
                (n_features, n_features),
                "%s w0" % self.cov_type,
            )
            precision_mat(self.cov_prior_, self.cov_type)
        elif self.cov_type == "diag":
            self.cov_prior_ = check_array(
                self.w0, dtype=[np.float64, np.float32], ensure_2d=False
            )
            check_param_shape(
                self.cov_prior_,
                (n_features,),
                "%s w0" % self.cov_type,
            )
            check_precision(self.cov_prior_, self.cov_type)
        elif self.w0 > 0.0:
            self.cov_prior_ = self.w0
        else:
            raise ValueError(
                "w0 should be greater than 0."
                % self.w0
            )

    def init(self, X, resp):

        nk, xk, sk = compute_parameters(
            X, resp, self.reg_covar, self.cov_type
        )

        #self._estimate_fractioninit(nk)
        self.compute_fractioninit(nk)
        self.compute_means(nk, xk)
        self.compute_precisions(nk, xk, sk)

    def compute_fractioninit(self, nk):
        if self.fraction_type == "dirichlet_process":
            self.fraction_concentration_ = (
                1.0 + nk,
                (
                        self.fraction_concentration_prior_
                        + np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))
                ),
            )
        else:
            self.fraction_concentration_ = self.fraction_concentration_prior_ + nk


    def compute_fractions(self, nk,A,B):

        if self.fraction_type == "dirichlet_process":
            self.fraction_concentration_ = (
                1.0 + nk,
                (
                        self.fraction_concentration_prior_
                        + np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))
                ),
            )
        else:
            self.fraction_concentration_ = [np.sum(self.fraction_concentration_)*np.sum(B * A[:, k]) for k in range(self.n_components)]


    def compute_means(self, nk, xk):
        self.mean_precision_ = self.mean_precision_prior_ + nk
        self.means_ = (self.mean_precision_prior_ * self.mean_prior_ + nk[:, np.newaxis] * xk
                      ) / self.mean_precision_[:, np.newaxis]

    def compute_precisions(self, nk, xk, sk):
        {
            "full": self.compute_wishart_f,
            "tied": self.compute_wishart_t,
            "diag": self.compute_wishart_d,
            "spherical": self.compute_wishart_s,
        }[self.cov_type](nk, xk, sk)

        self.precisions_cholesky_ = compute_precision(
            self.covs_, self.cov_type
        )

    def compute_wishart_f(self, nk, xk, sk):
        _, n_features = xk.shape

        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk#公式10.63

        self.covs_ = np.empty((self.n_components, n_features, n_features))

        for k in range(self.n_components):
            diff = xk[k] - self.mean_prior_
            self.covs_[k] = (
                    self.cov_prior_
                    + nk[k] * sk[k]
                    + nk[k]
                    * self.mean_precision_prior_
                    / self.mean_precision_[k]
                    * np.outer(diff, diff))

        self.covs_ /= self.degrees_of_freedom_[:, np.newaxis, np.newaxis]

    def compute_wishart_t(self, nk, xk, sk):
        _, n_features = xk.shape

        self.degrees_of_freedom_ = (
                self.degrees_of_freedom_prior_ + nk.sum() / self.n_components
        )

        diff = xk - self.mean_prior_
        self.covs_ = (
                self.cov_prior_
                + sk * nk.sum() / self.n_components
                + self.mean_precision_prior_
                / self.n_components
                * np.dot((nk / self.mean_precision_) * diff.T, diff)
        )


        self.covs_ /= self.degrees_of_freedom_

    def compute_wishart_d(self, nk, xk, sk):
        _, n_features = xk.shape

        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        diff = xk - self.mean_prior_
        self.covs_ = self.cov_prior_ + nk[:, np.newaxis] * (
                sk
                + (self.mean_precision_prior_ / self.mean_precision_)[:, np.newaxis]
                * np.square(diff)
        )

        self.covs_ /= self.degrees_of_freedom_[:, np.newaxis]

    def compute_wishart_s(self, nk, xk, sk):
        _, n_features = xk.shape

        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        diff = xk - self.mean_prior_
        self.covs_ = self.cov_prior_ + nk * (
                sk
                + self.mean_precision_prior_
                / self.mean_precision_
                * np.mean(np.square(diff), 1)
        )


        self.covs_ /= self.degrees_of_freedom_

    def _m_step(self, X,c, log_resp):
        n_samples, _ = X.shape

        nk, xk, sk = compute_parameters(
            X, np.exp(log_resp), self.reg_covar, self.cov_type
        )
        #print("means：")
        #print(self.means_)
        #print("covs：")
        #print(self.covs_)

        B = np.array([0.3, 0.2,0.5])

        model = Model()
        x = model.addMVar((len(B), self.n_components), vtype=GRB.BINARY)
        model.update()
        y = model.addMVar(n_samples, lb=-GRB.INFINITY)
        for i in range(n_samples):
            model.addConstr(y[i] == sum(
                (1 / (np.sqrt(2 * math.pi*self.covs_[j])) * np.exp(
                    -1 / (2 * self.covs_[j]) * (X[i] - self.means_[j]) ** 2)) * B @ x[:, j] for j in
                range(self.n_components)) - c[i])
        model.setObjective(y @ y, sense=GRB.MINIMIZE)
        model.addConstrs(sum(x[i, k] for k in range(self.n_components)) == 1 for i in range(len(B)))
        model.optimize()
       # model.write("my2.lp")
        A = []
        for i in range(len(B)):
            A.append(x[i].x)
        A = np.matrix(A)
        mix=B*A
        print("means：")
        print(self.means_)
        print("covs：")
        #print(self.covs_)
        for i in range(len(self.covs_)):
            print(np.sqrt(self.covs_[i]))
        self.compute_fractions(nk, A, B)
        self.compute_means(nk, xk)
        self.compute_precisions(nk, xk, sk)

    def et_log_fractions(self):
        if self.fraction_type == "dirichlet_process":
            digamma_sum = digamma(
                self.fraction_concentration_[0] + self.fraction_concentration_[1]
            )
            digamma_a = digamma(self.fraction_concentration_[0])
            digamma_b = digamma(self.fraction_concentration_[1])
            return (
                    digamma_a
                    - digamma_sum
                    + np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1]))
            )
        else:
            return digamma(self.fraction_concentration_) - digamma(
                np.sum(self.fraction_concentration_)
            )

    def et_log_prob(self, X):
        _, n_features = X.shape
        log_gauss = compute_log_prob(
            X, self.means_, self.precisions_cholesky_, self.cov_type
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)

        log_lambda = n_features * np.log(2.0) + np.sum(
            digamma(
                0.5 * (self.degrees_of_freedom_ - np.arange(0, n_features)[:, np.newaxis])), 0, )  # 逗号

        return log_gauss + 0.5 * (log_lambda - n_features / self.mean_precision_)

    def compute_lower_bound(self, log_resp, log_prob_norm):

        (n_features,) = self.mean_prior_.shape

        log_det_precisions_chol = compute_log_det(
            self.precisions_cholesky_, self.cov_type, n_features
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)

        if self.cov_type == "tied":
            log_wishart = self.n_components * np.float64(
                compute_log_wishart(
                    self.degrees_of_freedom_, log_det_precisions_chol, n_features
                )
            )
        else:
            log_wishart = np.sum(
                compute_log_wishart(
                    self.degrees_of_freedom_, log_det_precisions_chol, n_features
                )
            )

        if self.fraction_type == "dirichlet_process":
            log_norm_fraction = -np.sum(
                betaln(self.fraction_concentration_[0], self.fraction_concentration_[1])
            )
        else:
            log_norm_fraction = compute_log_dirichlet(self.fraction_concentration_)

        return (
                -np.sum(np.exp(log_resp) * log_resp)
                - log_wishart
                - log_norm_fraction
                - 0.5 * n_features * np.sum(np.log(self.mean_precision_))
        )


    def get_parameters(self):
        return (
            self.fraction_concentration_,
            self.mean_precision_,
            self.means_,
            self.degrees_of_freedom_,
            self.covs_,
            self.precisions_cholesky_,
        )

    def set_parameters(self, params):
        (
            self.fraction_concentration_,
            self.mean_precision_,
            self.means_,
            self.degrees_of_freedom_,
            self.covs_,
            self.precisions_cholesky_,
        ) = params


        if self.fraction_type == "dirichlet_process":
            fraction_dirichlet_sum = (
                    self.fraction_concentration_[0] + self.fraction_concentration_[1]
            )
            tmp = self.fraction_concentration_[1] / fraction_dirichlet_sum
            self.fractions_ = (
                    self.fraction_concentration_[0]
                    / fraction_dirichlet_sum
                    * np.hstack((1, np.cumprod(tmp[:-1])))
            )
            self.fractions_ /= np.sum(self.fractions_)
        else:
            self.fractions_ = self.fraction_concentration_ / np.sum(
                self.fraction_concentration_
            )

        if self.cov_type == "full":
            self.precisions_ = np.array(
                [
                    np.dot(prec_chol, prec_chol.T)
                    for prec_chol in self.precisions_cholesky_
                ]
            )

        elif self.cov_type == "tied":
            self.precisions_ = np.dot(
                self.precisions_cholesky_, self.precisions_cholesky_.T
            )
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2


