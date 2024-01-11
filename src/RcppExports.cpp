// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// grad
List grad(arma::mat x, arma::mat A, arma::mat mu_u, arma::vec mu_s, arma::mat mu_v, std::string noiseType, arma::vec noiseParams, arma::cube trans_u, arma::cube n_randomSample);
RcppExport SEXP _cycvb_grad(SEXP xSEXP, SEXP ASEXP, SEXP mu_uSEXP, SEXP mu_sSEXP, SEXP mu_vSEXP, SEXP noiseTypeSEXP, SEXP noiseParamsSEXP, SEXP trans_uSEXP, SEXP n_randomSampleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type mu_u(mu_uSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu_s(mu_sSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type mu_v(mu_vSEXP);
    Rcpp::traits::input_parameter< std::string >::type noiseType(noiseTypeSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type noiseParams(noiseParamsSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type trans_u(trans_uSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type n_randomSample(n_randomSampleSEXP);
    rcpp_result_gen = Rcpp::wrap(grad(x, A, mu_u, mu_s, mu_v, noiseType, noiseParams, trans_u, n_randomSample));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_cycvb_grad", (DL_FUNC) &_cycvb_grad, 9},
    {NULL, NULL, 0}
};

RcppExport void R_init_cycvb(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}