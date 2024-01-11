#include <RcppArmadillo.h>
#include <vector>
#include <map>
#include <string>
#include <math.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;
using namespace std;

cube sigmoid(cube x){
  cube res = 1/(1+exp(-x));
  return res;
}
mat sigmoid_mat(mat x){
  mat res = 1/(1+exp(-x));
  return res;
}
// mat prob_A(mat A){
//   mat prob = sigmoid(A-lambda*log(-c0/c1));
//   return prob;
// }
// mat A_grad(mat A){
//   cube AGrad(p,p,K);
//   for (int i = 0; i < p; ++i) {
//     AGrad.slice(i) = pow(binConcrete.slice(i),2)/lambda*exp(-(A+trans_u.slice(i))/lambda)*((A>(-lambda*log(-c1/c0)-l))&(A<(-lambda*log((c1-1)/(1-c0))-trans_u.slice(i))))
//   }
//   return AGrad;
// }
// [[Rcpp::export]]
List grad(arma::mat x,
          arma::mat A,
          arma::mat mu_u,
          arma::vec mu_s,
          arma::mat mu_v,
          std::string noiseType,
          arma::vec noiseParams,
          arma::cube trans_u,
          arma::cube n_randomSample){
  int n = x.n_rows,p = x.n_cols,K = 1000;
  double lambda = 2.0/3.0,c0 = -0.1,c1=1.1,pai=0.1,mu0 = 0,sigma0=1,sigma=0.1;
  // cube u_randomSample = randu<cube>(p,p,K),n_randomSample = randn<cube>(p,p,K),trans_u = u_randomSample/(1-u_randomSample);
  mat mu = mu_u*diagmat(mu_s)*mu_v.t();
  cube binConcrete(p,p,K),sample_B(p,p,K);
  for(int i=0;i<K;++i){
    binConcrete.slice(i) = A+trans_u.slice(i);
    sample_B.slice(i) = sigma*n_randomSample.slice(i)+mu;
  }
  binConcrete = sigmoid(binConcrete/lambda);
  cube hardBinConcrete = clamp((c1-c0)*binConcrete+c0,0.0,1.0);
  cube MonteCarloSample = hardBinConcrete%sample_B;
  cube grad_core(p,p,K);
  cube diff,Jacobian,scale_diff;
  mat noiseParamsMat = repmat(noiseParams, 1, n);
  diff = eye<mat>(p,p)-MonteCarloSample.each_slice();
  Jacobian = diff;
  scale_diff = diff;
  Jacobian.each_slice([p](mat& slice) { slice = slice*inv(trans(slice)*slice+0.01 * eye<mat>(p,p)); });
  scale_diff.each_slice([noiseParams](mat& slice) { slice = slice.each_col()/noiseParams; });
  if (noiseType == "gaussian") {
    grad_core = n*Jacobian-scale_diff.each_slice()*(x.t()*x);
  } else if (noiseType == "t") {
    cube temp = diff;
    temp.each_slice([noiseParamsMat,x](mat& slice) { slice = slice%(noiseParamsMat + 1) * x.t()/(pow(slice*x.t(),2)+noiseParamsMat)*x; });
    grad_core = n*Jacobian-temp;
  } else if (noiseType == "gumbel") {
    cube temp = scale_diff;
    temp.each_slice([noiseParamsMat,x](mat& slice) { slice = (1-exp(-slice*x.t()))/noiseParamsMat*x; });
    grad_core = n*Jacobian-temp;
  } else if (noiseType == "laplace") {
    cube temp = diff;
    temp.each_slice([noiseParamsMat,x](mat& slice) { slice = sign(slice*x.t())/noiseParamsMat*x; });
    grad_core = n*Jacobian-temp;
  }
  // for (int i = 0; i < K; ++i) {
  //   arma::mat t = MonteCarloSample.slice(i);
  //   arma::mat diff = eye<mat>(p,p) - t;
  //   arma::mat denom = diff.t() * diff + 0.01 * eye<mat>(p,p);
  //   arma::mat scale_diff = diff.each_col()/noiseParams*x.t();
  //
  //   if (noiseType == "gaussian") {
  //     grad_core.slice(i) = n * diff * inv(denom) - scale_diff * x;
  //   } else if (noiseType == "t") {
  //     mat temp = pow(diff*x.t(),2);
  //     grad_core.slice(i) = n * diff * inv(denom) - (((diff.each_col()%(noiseParams + 1) * x.t())) / ( (temp.each_col()+noiseParams)))*x;
  //   } else if (noiseType == "gumbel") {
  //     mat temp = 1 - exp(-scale_diff);
  //     grad_core.slice(i) = n * diff * inv(denom) - temp.each_col()/noiseParams * x;
  //   } else if (noiseType == "laplace") {
  //     mat temp = sign(diff * x.t());
  //     grad_core.slice(i) = n * diff * inv(denom) - temp.each_col()/noiseParams * x;
  //   }
  // }
  mat probA = sigmoid_mat(A-lambda*log(-c0/c1));
  // cube gradCore= grad_core(A,mu,noiseType);
  cube AGrad = trans_u;
  AGrad.each_slice([A,lambda,c1,c0](mat& slice) { slice = exp(-(A+slice)/lambda)%((A>(-lambda*log(-c1/c0)-slice))%(A<(-lambda*log((c1-1)/(1-c0))-slice))); });;
  AGrad = pow(binConcrete,2)/lambda%AGrad;
  // for (int i = 0; i < K; ++i) {
  //   AGrad.slice(i) = pow(binConcrete.slice(i),2)/lambda%exp(-(A+trans_u.slice(i))/lambda)%((A>(-lambda*log(-c1/c0)-trans_u.slice(i)))%(A<(-lambda*log((c1-1)/(1-c0))-trans_u.slice(i))));
  // }
  mat KL_gradA = (log(probA/pai+1e-10)-log((1-probA)/(1-pai)+1e-10)+(log(sigma0/sigma)+(pow(sigma,2)+pow(mu-mu0,2))/(2*pow(sigma0,2))-0.5))%pow(probA,2)%exp(-A+lambda*log(-c0/c1));
  mat gradA = mean(grad_core % sample_B % AGrad, 2);
  // for (int i = 0; i < p; ++i) {
  //   gradA.col(i) = mean(grad_core.row(i) % sample_B.row(i) % AGrad.row(i), 2);  // Element-wise multiplication and then mean along rows
  // }
  gradA = gradA+KL_gradA;
  mat KL_gradmu = probA%(mu-mu0)/pow(sigma0,2);
  mat gradmu = mean(grad_core % hardBinConcrete, 2);
  gradmu= gradmu+KL_gradmu;
  mat grad_mu_u = gradmu*mu_v*diagmat(mu_s);
  mat grad_mu_v = gradmu.t()*mu_u*diagmat(mu_s);
  vec grad_mu_s(p);
  for(int i=0;i<p;++i){
    grad_mu_s(i) = accu(gradmu%((mu_u.col(i)*mu_v.col(i).t())));
  }
  vec gradNoiseParams(p);
  // if (noiseType == "gaussian") {
  //   gradNoiseParams(i) += -0.5*sum(y,1)/pow(noise_params,2);
  // } else if (noiseType == "t") {
  //   gradNoiseParams(i) += 0.5*sum(log(1 + square(y) / noise_params),1) +
  //     (noise_params + 1) / noise_params *
  //     sum(square(y) / (noiseParams + square(x)),1);
  // } else if (noiseType == "gumbel") {
  //   gradNoiseParams(i) += sum(y/pow(noise_params,2)*(1-exp(-y/noise_params)),1);
  // } else if (noiseType == "laplace") {
  //   gradNoiseParams(i) += sum(abs(y),1)/pow(noise_params,2);
  // }
  // cube y = (eye<mat>(p,p)-MonteCarloSample.each_slice())*x.t();
  cube y = MonteCarloSample.each_slice()*x.t();
  y = x.t()-y.each_slice();
  cube gradNoiseParamsCube = y;
  mat gradNoiseParamsMat;
  // Rcpp::Rcout<<"this place"<<endl;
  if (noiseType == "gaussian") {
    gradNoiseParamsCube.each_slice([noiseParamsMat](mat& slice) { slice = -0.5*pow(slice,2)/pow(noiseParamsMat,2); });
  } else if (noiseType == "t") {
    gradNoiseParamsCube.each_slice([noiseParamsMat](mat& slice) { slice = 0.5*log(1 + pow(slice,2) / noiseParamsMat) -0.5*
      (noiseParamsMat + 1) / noiseParamsMat %
      pow(slice,2) / (noiseParamsMat + pow(slice,2)); });
  } else if (noiseType == "gumbel") {
    gradNoiseParamsCube.each_slice([noiseParamsMat](mat& slice) { slice = -slice/pow(noiseParamsMat,2)%(1-exp(-(slice/noiseParamsMat))); });
  } else if (noiseType == "laplace") {
    gradNoiseParamsCube.each_slice([noiseParamsMat](mat& slice) { slice = -abs(slice)/pow(noiseParamsMat,2); });
  }
  // Rcpp::Rcout<<gradNoiseParamsCube.n_rows<<" "<<gradNoiseParamsCube.n_cols<<" "<<gradNoiseParamsCube.n_slices<<endl;
  gradNoiseParamsMat = mean(gradNoiseParamsCube,2);
  gradNoiseParams = sum(gradNoiseParamsMat,1);
  // for (int i = 0; i < K; ++i) {
  //   arma::mat t = MonteCarloSample.slice(i);
  //   arma::mat y = (eye<mat>(p,p) - t)*x.t();
  //   // arma::mat denom  = t * diff * diff.t() + 0.01 * diagmat(arma::ones<arma::vec>(p));
  //
  //   if (noiseType == "gaussian") {
  //     gradNoiseParams += -0.5*sum(pow(y,2),1)/pow(noiseParams,2);
  //   } else if (noiseType == "t") {
  //     mat temp = pow(y,2);
  //     gradNoiseParams += 0.5*sum(log(1 + temp.each_col() / noiseParams),1) -0.5*
  //       (noiseParams + 1) / noiseParams %
  //       sum(temp / (noiseParams + temp.each_col()),1);
  //   } else if (noiseType == "gumbel") {
  //     gradNoiseParams += -sum(y.each_col()/pow(noiseParams,2)%(1-exp(-(y.each_col()/noiseParams))),1);
  //   } else if (noiseType == "laplace") {
  //     gradNoiseParams += -sum(abs(y),1)/pow(noiseParams,2);
  //   }
  // }
  // gradNoiseParams /=K;
  if (noiseType == "gaussian") {
    gradNoiseParams += n/2.0/noiseParams;
  } else if (noiseType == "t") {
    vec temp1 = 0.5*(noiseParams+1);
    NumericVector temp_1 = NumericVector(temp1.begin(),temp1.end());
    vec temp2 = 0.5*noiseParams;
    NumericVector temp_2 = NumericVector(temp2.begin(),temp2.end());
    NumericVector temp_3 = digamma(temp_1)-digamma(temp_2);
    vec temp3 = as<vec>(temp_3);
    gradNoiseParams += -0.5*n*(temp3-1.0/noiseParams);
  } else if (noiseType == "gumbel") {
    gradNoiseParams += n/noiseParams;
  } else if (noiseType == "laplace") {
    gradNoiseParams += n/noiseParams;
  }
  List resList;
  resList["grad_a"] = gradA;
  // resList["grad_mu"] = gradmu;
  resList["grad_mu.u"] = grad_mu_u;
  resList["grad_mu.s"] = grad_mu_s;
  resList["grad_mu.v"] = grad_mu_v;
  resList["grad_noise.params"] = gradNoiseParams;
  return resList;
  // return gradmu;
  // mat grad_mu_u = gradmu*mu_v*diagmat(mu_s);
  // return gradA;
  // return grad_core;
  // return MonteCarloSample;
  // List res;
  // res["urs"] = u_randomSample;
  // res["nrs"] = n_randomSample;
  // return res;
}

// NumericVector computeDigamma(NumericVector x) {
//   // Use Rcpp::digamma from Rcpp to compute the digamma function
//   NumericVector result = Rcpp::digamma(x);
//
//   return result;
// }
