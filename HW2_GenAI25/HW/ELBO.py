import torch
# ---------------- Loss ----------------
def _negative_log_likelihood(x, recon_mu):
    # flatten
    x = x.view(x.size(0), -1)
    recon_mu = recon_mu.view(recon_mu.size(0), -1)
    D = x.size(1)

    #--------------------------------------------------------------
    # To-DO: Compute the negative log likelihood (NLL) under Gaussian likelihood


    #--------------------------------------------------------------

    return nll

def _kl_diag_normal(mu, logvar):
    #--------------------------------------------------------------
    # To-DO: Compute the KL divergence between N(mu, var) and N(0, 1)

    #--------------------------------------------------------------

def ELBO(x,recon_mu, mu, logvar):

    nll = _negative_log_likelihood(x, recon_mu)
    kl = _kl_diag_normal(mu, logvar)
    #--------------------------------------------------------------
    # To-DO: Compute the negative ELBO loss

    #--------------------------------------------------------------
    return loss