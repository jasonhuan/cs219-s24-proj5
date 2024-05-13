#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <immintrin.h>
#include "./mp_solver.h"

#include <iostream>

#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <complex>


#define MAX_ITER 20
using namespace std;
extern __m128d _ZGVbN2v_sin(__m128d);
extern __m128d _ZGVbN2v_cos(__m128d);
extern void _ZGVbN2vvv_sincos(__m128d, __m128d, __m128d);

static int
channel_eq_func(const gsl_vector * x, void *data,
        gsl_vector * f)
{
  mp_config_t *mp_config = (mp_config_t *)data;

  double tau[MAX_NOF_PATHS];
  double nu[MAX_NOF_PATHS];
  _Complex double h[MAX_NOF_PATHS];

  for (int p = 0; p < mp_config->nof_paths; ++p) {
    tau[p] = gsl_vector_get(x, p);
    nu[p] = gsl_vector_get(x, mp_config->nof_paths+p);
    h[p] = gsl_vector_get(x, 2*mp_config->nof_paths+p) + gsl_vector_get(x, 3*mp_config->nof_paths+p)*I;
  }

  for (int k = 0; k < mp_config->nof_pilots; ++k)
  {
    _Complex double Yk = 0;
    for (int p = 0; p < mp_config->nof_paths; ++p) {
      Yk += h[p] * cexp(-I*2*M_PI*(tau[p] * (mp_config->m[k] + 0.5) - nu[p] * (mp_config->n[k] + 0.5)));
    }
    gsl_vector_set(f, k, cabs(Yk - mp_config->y[k]));
  }

  return GSL_SUCCESS;
}

//static int
//channel_eq_func_opt(const gsl_vector * x, void *data,
//        gsl_vector * f)
//{
//  mp_config_t *mp_config = (mp_config_t *)data;
//
//  double tau[MAX_NOF_PATHS];
//  double nu[MAX_NOF_PATHS];
//  double h_r[MAX_NOF_PATHS];
//  double h_i[MAX_NOF_PATHS];
//
//  for (int p = 0; p < mp_config->nof_paths; ++p) {
//    tau[p] = gsl_vector_get(x, p);
//    nu[p] = gsl_vector_get(x, mp_config->nof_paths+p);
//    h_r[p] = gsl_vector_get(x, 2*mp_config->nof_paths+p);
//    h_i[p] = gsl_vector_get(x, 3*mp_config->nof_paths+p);
//  }
//  
//  double tbuf[2] __attribute__ ((aligned (16)));
//
///*
//  __m128d vh_r = __mm_load_pd(h_r);
//  __m128d vh_i = __mm_load_pd(h_i);
//  __m128d vnu = __mm_load_pd(nu);
//  __m128d vtau = __mm_load_pd(tau);
//*/
//
//  for (int k = 0; k < mp_config->nof_pilots; k++)
//  {
//    __m128d vm = _mm_set_pd1(-(mp_config->m[k] + 0.5) * 2 * M_PI);
//    __m128d vn = _mm_set_pd1(-(mp_config->n[k] + 0.5) * 2 * M_PI);
//
//    for (int p = 0; p < mp_config->nof_paths; p += 2)
//    {
//      __m128d vtau = _mm_load_pd(&tau[p]);
//      __m128d vnu = _mm_load_pd(&nu[p]);
//
//      __m128d v1 = _mm_mul_pd(vm, vtau);
//      __m128d v2 = _mm_mul_pd(vn, vnu);
//
//      __m128d vangle = _mm_sub_pd(v1, v2);
//
//#ifdef USE_SINCOS
//      // DOES NOT WORK`
//      __m128d vreal_mem;
//      __m128d vimag_mem;
//      _ZGVbN2vvv_sincos(vangle, vimag_mem, vreal_mem);
//      __m128d vreal = _mm_load_pd((double *)&vreal_mem);
//      __m128d vimag = _mm_load_pd((double *)&vimag_mem);
//#else
//      __m128d vreal = _ZGVbN2v_cos(vangle);
//      __m128d vimag = _ZGVbN2v_sin(vangle);
//#endif
//
//      __m128d vh_r = _mm_load_pd(&h_r[p]);
//      __m128d vh_i = _mm_load_pd(&h_i[p]);
//
//      __m128d vYk_r = _mm_sub_pd(_mm_mul_pd(vreal, vh_r), _mm_mul_pd(vimag, vh_i));
//      __m128d vYk_i = _mm_add_pd(_mm_mul_pd(vreal, vh_i), _mm_mul_pd(vimag, vh_r));
//      __m128d vYk = _mm_hadd_pd(vYk_r, vYk_i); // Real at [63:0], imag at [127:64]
//
//      __m128d vy = _mm_set_pd(cimag(mp_config->y[k]), creal(mp_config->y[k]));
//
//      __m128d vdiff = _mm_sub_pd(vy, vYk);
//      __m128d vsq = _mm_mul_pd(vdiff, vdiff);
//
//      __m128d vres = _mm_hadd_pd(vsq, vsq);
//      _mm_store_pd(tbuf, vres);
//      gsl_vector_set(f, k, sqrt(tbuf[0]));
//    }
//  }
//
//  return GSL_SUCCESS;
//}

int mp_solver(mp_config_t *mp_config, mp_profile_t *mp_profile)
{
  const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
  gsl_multifit_nlinear_workspace *w;
  gsl_multifit_nlinear_fdf fdf;
  gsl_multifit_nlinear_parameters fdf_params =
    gsl_multifit_nlinear_default_parameters();

  double weights[MAX_NOF_PILOTS];
  double x_init[4*MAX_NOF_PATHS];
  for (int p = 0; p < mp_config->nof_paths; ++p) {
    x_init[p] = mp_profile->tau[p];
    x_init[mp_config->nof_paths+p] = mp_profile->nu[p];
    x_init[2*mp_config->nof_paths+p] = creal(mp_profile->h[p]);
    x_init[3*mp_config->nof_paths+p] = cimag(mp_profile->h[p]);
  }

  gsl_vector_view x = gsl_vector_view_array(x_init, 4*mp_config->nof_paths);
  gsl_vector_view wts = gsl_vector_view_array(weights, mp_config->nof_pilots);
  int status, info;

  const double xtol = 1e-6;
  const double gtol = 1e-6;
  const double ftol = 0.0;

  /* define the function to be minimized */
  fdf.f = channel_eq_func; //changed
  fdf.df = NULL;   /* set to NULL for finite-difference Jacobian */
  fdf.fvv = NULL;     /* not using geodesic acceleration */
  fdf.n = mp_config->nof_pilots;
  fdf.p = 4*mp_config->nof_paths;
  fdf.params = mp_config;

  /* this is the data to be fitted */
  for (int i = 0; i < mp_config->nof_pilots; ++i){
    weights[i] = 1.0;
  };

  /* allocate workspace with default parameters */
  w = gsl_multifit_nlinear_alloc(T, &fdf_params, mp_config->nof_pilots, 4*mp_config->nof_paths);

  /* initialize solver with starting point and weights */
  gsl_multifit_nlinear_winit(&x.vector, &wts.vector, &fdf, w);

  /* solve the system with a maximum of 100 iterations */
  status = gsl_multifit_nlinear_driver(MAX_ITER, xtol, gtol, ftol,
                                       NULL, NULL, &info, w);

  /* read final solution */
  gsl_vector *res = gsl_multifit_nlinear_position(w);
  for (int p = 0; p < mp_config->nof_paths; ++p) {
    mp_profile->tau[p] = gsl_vector_get(res, p);
    mp_profile->nu[p] = gsl_vector_get(res, mp_config->nof_paths+p);
    double h_r = gsl_vector_get(res, 2*mp_config->nof_paths+p);
    double h_i = gsl_vector_get(res, 3*mp_config->nof_paths+p);
    mp_profile->h[p] = h_r + h_i*I;
  }

  gsl_multifit_nlinear_free(w);

  return 0;
}

int main() {
    // read input
    // mp_config_t mp_config;
    // mp_profile_t mp_profile;
    // mp_config.nof_pilots = 32;
    // mp_config.nof_paths = 3;


    std::ifstream file("sample_input.csv");
    if (!file.is_open()) {
      std::cout << "Failed to open the file." << std::endl;
      return 1; // Exit the program
    }

    std::string line;
    std::vector<std::vector<std::string>> data;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> row;
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    // Print the imported data
    // for (const auto& row : data) {
    //     for (const auto& cell : row) {
    //         std::cout << cell << " ";
    //     }
    //     std::cout << std::endl;
    // }

    for (const auto& row : data) {
      mp_config_t mp_config;
      mp_profile_t mp_profile;
      mp_config.nof_pilots = 32;
      mp_config.nof_paths = 3;
      
      int arr[MAX_NOF_PILOTS];
      arr[0] = stoi(row[0]);
      mp_config.m[0] = stoi(row[0]);

      arr[0] = stoi(row[1]);
      mp_config.n[0] = stoi(row[1]);

      // std::cout << "mpconfig m" << mp_config.m[0] << std::endl;
      _Complex double h;

      std::string complexStr = row[2]; // Complex number as a string
      std::cout << complexStr << std::endl;
      
      _Complex double complexNum;
      //std::istringstream iss(complexStr);
      
      //rewrite this line
      //iss >> complexNum; // Extract complex number from string
      

      //will work when above line is finished
      //mp_config.y[0] = complexNum;

      break;

    }
    std::cout << "main completed" << std::endl;
}
