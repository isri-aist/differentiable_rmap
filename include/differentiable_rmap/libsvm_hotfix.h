// This file contains the code copied and edited from LIBSVM (https://www.csie.ntu.edu.tw/~cjlin/libsvm/).
// Please refer to the license of LIBSVM below.

/*
 *  Copyright (c) 2000-2021 Chih-Chung Chang and Chih-Jen Lin
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *  1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 *
 *  3. Neither name of copyright holders nor the names of its contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 *  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

static const char * svm_type_table[] = {"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr", NULL};

static const char * kernel_type_table[] = {"linear", "polynomial", "rbf", "sigmoid", "precomputed", NULL};

inline int svm_save_model_hotfix(const char * model_file_name, const svm_model * model)
{
  FILE * fp = fopen(model_file_name, "w");
  if(fp == NULL) return -1;

  char * old_locale = setlocale(LC_ALL, NULL);
  if(old_locale)
  {
    old_locale = strdup(old_locale);
  }
  setlocale(LC_ALL, "C");

  const svm_parameter & param = model->param;

  fprintf(fp, "svm_type %s\n", svm_type_table[param.svm_type]);
  fprintf(fp, "kernel_type %s\n", kernel_type_table[param.kernel_type]);

  if(param.kernel_type == POLY) fprintf(fp, "degree %d\n", param.degree);

  if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
    fprintf(fp, "gamma %.17g\n", param.gamma);

  if(param.kernel_type == POLY || param.kernel_type == SIGMOID) fprintf(fp, "coef0 %.17g\n", param.coef0);

  int nr_class = model->nr_class;
  int l = model->l;
  fprintf(fp, "nr_class %d\n", nr_class);
  fprintf(fp, "total_sv %d\n", l);

  {
    fprintf(fp, "rho");
    for(int i = 0; i < nr_class * (nr_class - 1) / 2; i++) fprintf(fp, " %.17g", model->rho[i]);
    fprintf(fp, "\n");
  }

  if(model->label)
  {
    fprintf(fp, "label");
    for(int i = 0; i < nr_class; i++) fprintf(fp, " %d", model->label[i]);
    fprintf(fp, "\n");
  }

  if(model->probA) // regression has probA only
  {
    fprintf(fp, "probA");
    for(int i = 0; i < nr_class * (nr_class - 1) / 2; i++) fprintf(fp, " %.17g", model->probA[i]);
    fprintf(fp, "\n");
  }
  if(model->probB)
  {
    fprintf(fp, "probB");
    for(int i = 0; i < nr_class * (nr_class - 1) / 2; i++) fprintf(fp, " %.17g", model->probB[i]);
    fprintf(fp, "\n");
  }

  // hotfix: Since an invalid value is set in model->nSV, set the value manually only for two-class classification.
  if(param.svm_type == C_SVC || param.svm_type == NU_SVC)
  {
    if(nr_class != 2) throw std::runtime_error("multiple classes are not supported in svm_save_model_hotfix");
    fprintf(fp, "nr_sv");
    for(int i = 0; i < nr_class; i++)
      if(i == 0)
        fprintf(fp, " %d", l);
      else
        fprintf(fp, " 0");
    fprintf(fp, "\n");
  }

  fprintf(fp, "SV\n");
  const double * const * sv_coef = model->sv_coef;
  const svm_node * const * SV = model->SV;

  for(int i = 0; i < l; i++)
  {
    for(int j = 0; j < nr_class - 1; j++) fprintf(fp, "%.17g ", sv_coef[j][i]);

    const svm_node * p = SV[i];

    if(param.kernel_type == PRECOMPUTED)
      fprintf(fp, "0:%d ", (int)(p->value));
    else
      while(p->index != -1)
      {
        fprintf(fp, "%d:%.8g ", p->index, p->value);
        p++;
      }
    fprintf(fp, "\n");
  }

  setlocale(LC_ALL, old_locale);
  free(old_locale);

  if(ferror(fp) != 0 || fclose(fp) != 0)
    return -1;
  else
    return 0;
}
