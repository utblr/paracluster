#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <math.h>

int
main (void)
{
  double data[] = { 0,   0.8, 0.6, 0.1, 0,   0,
                    0.8, 0,   0.9, 0,   0,   0,
                    0.6, 0.9,   0, 0,   0,   0.2,
                    0.1, 0,     0, 0,   0.6, 0.7,
                    0,   0,     0, 0.6, 0,   0.8,
		    0,   0,   0.2, 0.7, 0.8, 0};

  int n = pow(sizeof(data)/sizeof(double),0.5);
  
  gsl_matrix_view m
    = gsl_matrix_view_array (data, n, n);

  gsl_vector *eval = gsl_vector_alloc (n);
  gsl_matrix *evec = gsl_matrix_alloc (n, n);

  gsl_eigen_symmv_workspace * w =
    gsl_eigen_symmv_alloc (n);

  gsl_eigen_symmv (&m.matrix, eval, evec, w);

  gsl_eigen_symmv_free (w);

  gsl_eigen_symmv_sort (eval, evec,
                        GSL_EIGEN_SORT_ABS_ASC);

  {
    int i;

    for (i = 0; i < n; i++)
      {
        double eval_i
           = gsl_vector_get (eval, i);
        gsl_vector_view evec_i
           = gsl_matrix_column (evec, i);

        printf ("eigenvalue = %g\n", eval_i);
        printf ("eigenvector = \n");
        gsl_vector_fprintf (stdout,
                            &evec_i.vector, "%g");
      }
  }

  gsl_vector_free (eval);
  gsl_matrix_free (evec);

  return 0;
}
