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
#include "thread_pool.h"
#include "./mp_solver.h"
//#include "/usr/local/include/gpufit/gpufit.h"
#include "/home/shared/Gpufit/Gpufit/gpufit.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <complex>
#include <cstring>
#include <chrono>
#include <thread>
#include <future>
#include <cassert>
#include <cstdlib>

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

  for (int p = 0; p < mp_config->nof_paths; ++p) 
  {
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

void test_gpufit(std::vector<float> _initial_parameters, std::vector<float> _initial_data, std::vector<float> _output_parameters)
{
	/*
		This example demonstrates a simple, minimal program containing all 
		of the required parameters for a call to the Gpufit function.  The example 
		can be built and executed within the project environment. Please note that 
		this code does not actually do anything other than make a single call to 
		gpufit().

		In the first section of the code, the *model ID* is set, memory space for 
		initial parameters and data values is allocated, the *fit tolerance* is set, 
		the *maximum number of iterations* is set, the *estimator ID* is set, and 
		the *parameters to fit array* is initialized.  Note that in most applications, 
		the data array will already exist and it will be unnecessary to allocate 
		additional space for data.  In this example, the *parameters to fit* array 
		is initialized to all ones, indicating that all model parameters should be 
		adjusted in the fit.
	*/

    /*************** definition of input and output parameters  ***************/

	// number of fits, number of points per fit
	std::size_t const n_fits = 10; // default values
	std::size_t const n_points_per_fit = 10;

	// model ID and number of model parameters
	int const model_id = GAUSS_1D;
	std::size_t const n_model_parameters = 3*3; //hardcoded as 4*mp_config->nof_paths

	// initial parameters
	std::vector< REAL > initial_parameters(*_initial_parameters); //replace: initial paths (data struct placeholder): n_fits * n_model_parameters

	// data
	std::vector< REAL > data(*_initial_data); //replace: .csv input values: n_points_per_fit * n_fits

	// tolerance
	REAL const tolerance = 0.001f;

	// maximum number of iterations
	int const max_number_iterations = 100; // same from previous (GSL jacobian curve fitting)

	// estimator ID
	int const estimator_id = LSE;

	// parameters to fit (all of them)
	std::vector< int > parameters_to_fit(n_model_parameters, 1);

	// output parameters
	std::vector< REAL > output_parameters(*_output_parameters); //replace: output _Complex variable array: n_fits * n_model_parameters
	std::vector< int > output_states(n_fits);
	std::vector< REAL > output_chi_square(n_fits);
	std::vector< int > output_number_iterations(n_fits);

    /***************************** call to gpufit  ****************************/

	int const status = gpufit
        (
            n_fits,
            n_points_per_fit,
            data.data(),
            0,
            model_id,
            initial_parameters.data(),
            tolerance,
            max_number_iterations,
            parameters_to_fit.data(),
            estimator_id,
            0,
            0,
            output_parameters.data(),
            output_states.data(),
            output_chi_square.data(),
            output_number_iterations.data()
        );

    /****************************** status check  *****************************/

	if (status != ReturnState::OK)
	{
		throw std::runtime_error(gpufit_get_last_error());
	} else {
		std::cout << "working gpufit_test()\n";
	}
}

int main(int argc, char *argv[]) {
    assert(("You should pass in at least 1 argument (true/false) indicating multithreaded version or not.", argc >= 2 ));
    assert(("Ensure that the argument is either true or false", std::string(argv[1]) == "true" || std::string(argv[1]) == "false"));

    // true indicates thread pooled version, false indicates single threaded
    bool multithread; 
    istringstream(argv[1]) >> std::boolalpha >> multithread;
    
    int num_threads;
    
    if (multithread) {
        assert(("You should pass in a second argument to indicate how many threads should be used", argc == 3));
        std::stringstream ss(argv[2]);
        int value;
        if (ss >> value) {
            std::cout << "Argument can be cast to an integer: " << value << std::endl;
            assert(("Value must be a positive integer", value > 0));
            num_threads = value;
        } else {
            std::cerr << "Invalid argument: not an integer" << std::endl;
            exit(1);
        }
    } 
    cout << "num_threads: " << num_threads << std::endl;


    auto start = std::chrono::high_resolution_clock::now();
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

    // ==================STARTING for loop=====================
    cout << "==================STARTING for loop=====================" << '\n';

    mp_config_t mp_config;
    mp_config.nof_pilots = MAX_NOF_PILOTS;
    mp_config.nof_paths = MAX_NOF_PATHS;
    mp_profile_t mp_profile;
    int i = 0;
    for (const auto& row : data) {
      cout << "===================================loop iteration: " << i << '\n';
      
      mp_config.m[i] = stoi(row[0]);
      cout << "mp_config.m: " << mp_config.m[i] << '\n';

      mp_config.n[i] = stoi(row[1]);
      cout << "mp_config.n: " << mp_config.n[i] << '\n';

      // ==================STARTING _Complex=====================
      cout << "==================STARTING _Complex=====================" << '\n';
      // std::cout << "mpconfig m" << mp_config.m[0] << std::endl;
      _Complex double h;

      std::string complexStr = row[2]; // Complex number as a string
      std::cout << complexStr << std::endl; //DEBUG0
      
      _Complex double complexNum;
     
      string complexNum_real = complexStr.substr(0,complexStr.find('+'));
      string complexNum_imagin = complexStr.substr(complexStr.find('+') + 1, complexStr.length() - complexNum_real.length() - 2); // minus 2 because one to take off the 'j' and one to make up for starting the remaining substring 1 character AFTER the plus sign (instead of on it)

      cout << "test complexNum_real: " << complexNum_real << '\n' << "test complexNum_imagin: " << complexNum_imagin << '\n'; //DEBUG1 

      double complexNum_real_d = stod(complexNum_real);
      double complexNum_imagin_d = stod(complexNum_imagin);

      double a[2];
      a[0] = complexNum_real_d;
      a[1] = complexNum_imagin_d;
      memcpy(&complexNum, a, sizeof complexNum);

      //memcpy(&complexNum, &complexNum_real_d, sizeof complexNum_real_d);
      //memcpy(&complexNum + sizeof complexNum_real_d, &complexNum_imagin_d, sizeof complexNum_imagin_d);
      
      cout << "real: " << creal(complexNum) << '\n' << "imaginary: " << cimag(complexNum) << '\n'; //DEBUG2
 
      //will work when above line is finished
      mp_config.y[i] = complexNum;
      cout << "mp_config real: " << creal(mp_config.y[i]) << '\n' << "mp_config imaginary: " << cimag(mp_config.y[i]) << '\n'; //DEBUG2

      i++;
    }
    // ==================AFTER for loop=====================
    cout << "==================AFTER for loop=====================" << '\n';

    cout << "==================GPUFIT testing=====================" << '\n';
    int num_fits; 
    std::vector<float> initial_parameters;
    std::vector<float> initial_data;
    std::vector<float> output_parameters;

    // initialize parameter array
    for(int i = 0; i < num_fits*3*MAX_NOF_PATHS; i++){
      initial_parameters.push_back(0); //initalize parameters as all 0's
    }

    for(int i = 0; i < num_fits*MAX_NOF_PILOTS; i++){
      initial_data.push_back(float(rand())); // randomize data for input
    }

    for(int i = 0; i < num_fits*3*MAX_NOF_PATHS; i++){
      output_parameters.push_back(0);
    }

    test_gpufit(initial_parameters, initial_data, output_parameters); //test gpufit



    cout << "NOF_PATHS: " << mp_config.nof_paths <<  "\n";
    cout << "NOF_PILOTS: " << mp_config.nof_pilots <<  "\n";
    std::cout << "main completed" << std::endl;

  /*
  int num_iterations = 2000;
  if (!multithread) {
    
    cout << "=================begin singlethreaded" << num_iterations <<  "x mp_solver() and timing=================" << "\n";
    for (int i = 0; i < num_iterations; ++i) {
      mp_solver(&mp_config, &mp_profile);
    }
  } else {
    ThreadPool pool(num_threads);

    std::vector<std::future<int>> results;
    cout << "=================begin calling mp_solver() and timing=================" << "\n";
    try {
      for (int i = 0; i < num_iterations; i++) {
        results.emplace_back(pool.enqueue(mp_solver, &mp_config, &mp_profile));
      }
      for (auto&& result : results) {
        result.get();
      }
    } catch (const std::system_error &e) {
      std:cout << "Error creating thread: " << e.what() << std::endl;
    }
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  cout << "Elapsed time: " << duration.count() << " us\n";
  cout << "=================done calling mp_solver() and timing=================" << "\n";
  */
}

