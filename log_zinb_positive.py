import torch
import torch.nn as nn
def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    """
    Adopted from: https://github.com/YosefLab/scVI/blob/master/scvi/models/log_likelihood.py#L11
    Equations follow the paper: https://www.nature.com/articles/s41467-017-02554-5

    Parameters
    ----------
    mu: tensor (nsamples, nfeatures)
        Mean of the negative binomial (has to be positive support).
    theta: tensor (nsamples, nfeatures)
        Inverse dispersion parameter (has to be positive support).
    pi: tensor (n_samples, nfeatures)
        Logit of the dropout parameter (real support).
    eps: numeric
        Numerical stability constant.
    """

    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = - pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = - softplus_pi + \
        pi_theta_log + \
        x * (torch.log(mu + eps) - log_theta_mu_eps) + \
        torch.lgamma(x + theta) - \
        torch.lgamma(theta) - \
        torch.lgamma(x + 1)
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return torch.sum(res, dim=-1)
