# rm(list = ls())
# sourceCpp("./code/02-2grad_test.cpp")

cycvb <- function(x,noise.type=c("gaussian", "t", "gumbel","laplace"),seed = 1){
  noise.type = match.arg(noise.type)
  n = nrow(x)
  p = ncol(x)
  ########initialization##########
  #number of Monte Carlo samples
  set.seed(seed)
  K = 1000
  #parameters and random uniform samples for hard concrete distribution
  lambda = 2/3
  u = array(runif(K*p^2),dim = c(p,p,K))
  l = u/(1-u)
  # a = array(0,dim = c(p,p,K))
  # s = sigmoid(((a)+l)/lambda)
  c0 = -0.1
  c1 = 1.1
  # sh = s*(c1-c0)+c0
  # z = apply(sh, c(1,2,3), function(x){min(1,max(0,x))})
  # # hist(z)
  # #parameters and random samples for normal distribution
  ub = array(rnorm(K*p^2),dim = c(p,p,K))
  # mu= array(0,dim = c(p,p,K))
  # # mu = t(gnlearn::pc(as.data.frame(x),to = "adjacency"))
  # # mu = array(rep(mu,K),dim = c(p,p,K))
  # sigma= array(1,dim = c(p,p,K))
  #parameters of noise distribution
  noise.params = rep(0.5,p)
  at = matrix(-2,p,p)
  param = list()
  param$a=at
  mu0 = matrix(0,p,p)
  # mu0[testn!=0] = runif(length(which(testn!=0)),-0.8,0.8)
  # mu0 = B0
  svd0=svd(mu0)
  param$mu.u=svd0$u
  param$mu.s =svd0$d
  param$mu.v=svd0$v

  param$noise.params = noise.params
  # param[(1+2*p^2):(3*p^2)] = sigma[,,1]
  upper = matrix(100,p,p)
  upper2u = matrix(100,p,p)
  upper2s = rep(1,p)
  upper2v = matrix(100,p,p)
  diag(upper) = -10
  upper3 = rep(100,p)
  lower = -matrix(100,p,p)
  lower2u = -matrix(100,p,p)
  lower2s = -rep(1,p)
  lower2v = -matrix(100,p,p)
  lower3 = rep(0.0001,p)
  upper = list(upper_a=upper,upper_mu.u=upper2u,upper_mu.s=upper2s,upper_mu.v=upper2v,upper_noise.params=upper3)
  lower = list(lower_a=lower,lower_mu.u=lower2u,lower_mu.s=lower2s,lower_mu.v=lower2v,lower_noise.params=lower3)

  stepsize = list()
  stepsize$a = 0.001
  stepsize$mu.u = 0.001
  stepsize$mu.s = 0.001
  stepsize$mu.v = 0.001
  stepsize$noise.params = 0.0001

  #############iteration################
  max_iter = 500
  # pb <- progress_bar$new(format = "  complete [:bar] :percent eta: :eta",
                         # total = max_iter, clear = FALSE, width= 60)
  for (iter in 1:max_iter) {
    grad_all = grad(x,param$a,param$mu.u,param$mu.s,param$mu.v,noise.type,param$noise.params,l,ub)
    grad_a = grad_all$grad_a
    grad_mu.s = grad_all$grad_mu.s
    grad_noise.params = grad_all$grad_noise.params
    grad_mu.u = grad_all$grad_mu.u
    grad_mu.v = grad_all$grad_mu.v
    mu.utemp = param$mu.u
    mu.vtemp = param$mu.v
    proj_mu.u = (diag(p)-mu.utemp%*%t(mu.utemp))%*%grad_mu.u+0.5*mu.utemp%*%(t(mu.utemp)%*%grad_mu.u-t(grad_mu.u)%*%mu.utemp)
    proj_mu.v = (diag(p)-mu.vtemp%*%t(mu.vtemp))%*%grad_mu.v+0.5*mu.vtemp%*%(t(mu.vtemp)%*%grad_mu.v-t(grad_mu.v)%*%mu.vtemp)
    # grad_all$grad_mu.u = proj_mu.u
    # grad_all$grad_mu.v = proj_mu.v
    param$a = pmin(upper$upper_a,pmax(lower$lower_a,param$a-stepsize$a*grad_a))
    param$mu.s = pmin(upper$upper_mu.s,pmax(lower$lower_mu.s,param$mu.s-stepsize$mu.s*grad_mu.s))
    param$noise.params = pmin(upper$upper_noise.params,pmax(lower$lower_noise.params,param$noise.params-stepsize$noise.params*grad_noise.params))
    param$mu.u = pmin(upper$upper_mu.u,pmax(lower$lower_mu.u,param$mu.u-stepsize$mu.u*proj_mu.u))
    param$mu.v = pmin(upper$upper_mu.v,pmax(lower$lower_mu.v,param$mu.v-stepsize$mu.v*proj_mu.v))
    # param = pmin(upper,pmax(lower,param-stepsize*gparam))
    # param$noise.params = noise.params
    param$mu.u = qr.Q(qr(param$mu.u))
    param$mu.v = qr.Q(qr(param$mu.v))
    # pb$tick()
    # Sys.sleep(1/100)
  }
  est_a = param$a
  est_prob = sigmoid((est_a)-lambda*log(-c0/c1))
  est_mu.u = param$mu.u
  est_mu.s = param$mu.s
  est_mu.v = param$mu.v
  est_mu = est_mu.u%*%diag(est_mu.s)%*%t(est_mu.v)
  est_noise.params = param$noise.params
  return(list(est_a = est_a,est_prob = est_prob,est_mu = est_mu,est_noise.params = est_noise.params))
}

sigmoid <-function(x){
  res = 1/(1+exp(-x))
  return(res)
}

