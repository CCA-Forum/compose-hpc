#include <stdio.h>
#include <stdlib.h>
typedef int bool;
int nbfn;
int ichunk;
int g_counter;
int icut1;
int icut2;
int icut3;
int icut4;
int g_fock;
int newtask;
int maxnbfn;
int g_schwarz;
int g_dens;
int tol2e;
int GA_Nodeid();
int next_4chunk(int *,int *,int *,int *,int *,int *);
void GA_Zero(int );
long gettask();
int translate_task(long ,int *,int *,int *,int *,int *,int *);
bool is_task_local(int ,int *,int *,int ,int *,int *);
void NGA_Get(int ,int *,int *,double (*)[10UL],int *);
void clean_chunk(double (*)[10UL]);
void g(double *,int ,int ,int ,int );
void NGA_Acc(int ,int *,int *,double (*)[10UL],int *,double *);
long NGA_Read_inc(int ,int *,int );
int NGA_Locate_region(int ,int *,int *,int *,int *);

void twoel(double *schwmax,double *etwo)
{
  double f_ij[ichunk][ichunk];
  double d_kl[ichunk][ichunk];
  double f_ik[ichunk][ichunk];
  double d_jl[ichunk][ichunk];
  double s_ij[ichunk][ichunk];
  double s_kl[ichunk][ichunk];
  double one;
  long long ijcnt;
  long long klcnt;
  long long ijklcnt;
  int lo[4UL];
  int hi[4UL];
  int lo_ik[2UL];
  int hi_ik[2UL];
  int lo_jl[2UL];
  int hi_jl[2UL];
  int i;
  int j;
  int k;
  int l;
  int iloc;
  int jloc;
  int kloc;
  int lloc;
  int ld;
  int ich;
  int it;
  int jt;
  int kt;
  int lt;
  int dotask;
  long taskid;
  int accum;
  int itask;
  double gg;
  one = 1.00;
  ijcnt = icut1;
  klcnt = icut2;
  ijklcnt = icut3;
  GA_Zero(g_counter);
  ld = maxnbfn;
  ich = ichunk;
/*% TASCEL version=1 */
  taskid = gettask();
/*% TASCEL version=1 */ 
  dotask = translate_task(taskid, lo, hi, &it, &jt, &kt, &lt);
  itask = 0;
  newtask = 1;
  accum = 0;
/*% TASCEL version=1 */
  while(dotask != 0){
    lo_ik[0] = lo[0];
    lo_ik[1] = lo[2];
    hi_ik[0] = hi[0];
    hi_ik[1] = hi[2];
    lo_jl[0] = lo[1];
    lo_jl[1] = lo[3];
    hi_jl[0] = hi[1];
    hi_jl[1] = hi[3];
    NGA_Get(g_schwarz,lo,hi,s_ij,&ich);
    if (!is_task_real(s_ij, *schwmax, lo, hi))
      continue; 
    NGA_Get(g_schwarz,(lo + 2),(hi + 2),s_kl,&ich);
    NGA_Get(g_dens,(lo + 2),(hi + 2),d_kl,&ich);
    NGA_Get(g_dens,lo_jl,hi_jl,d_jl,&ich);
    itask = (itask + 1);
    clean_chunk(f_ij);
    clean_chunk(f_ik);
    for (i = lo[0]; i <= hi[0]; i++) {
      iloc = (i - lo[0]);
      for (j = lo[1]; j <= hi[1]; j++) {
        jloc = (j - lo[1]);
        if ((s_ij[iloc][jloc] *  *schwmax) < tol2e) {
          icut1 = (icut1 + (((hi[2] - lo[2]) + 1) * ((hi[3] - lo[3]) + 1)));
        }
        else {
          for (k = lo[2]; k <= hi[2]; k++) {
            kloc = (k - lo[2]);
            for (l = lo[3]; l <= hi[3]; l++) {
              lloc = (l - lo[3]);
              if ((s_ij[iloc][jloc] * s_kl[kloc][lloc]) < tol2e) {
                icut2 = (icut2 + 1);
              }
              else {
                g(&gg,i,j,k,l);
                f_ij[iloc][jloc] = (f_ij[iloc][jloc] + (gg * d_kl[kloc][lloc]));
                f_ik[iloc][kloc] = (f_ik[iloc][kloc] - ((0.50 * gg) * d_jl[jloc][lloc]));
                icut3 = (icut3 + 1);
                accum = 1;
              }
            }
          }
        }
      }
    }
    if (accum != 0) {
      NGA_Acc(g_fock,lo,hi,f_ij,&ich,&one);
      NGA_Acc(g_fock,lo_ik,hi_ik,f_ik,&ich,&one);
    }
    taskid = gettask(); 
    dotask = translate_task(taskid, lo, hi, &it, &jt, &kt, &lt);
    if (dotask != 0) 
      accum = 0;
  }
  ijcnt = (icut1 - ijcnt);
  klcnt = (icut2 - klcnt);
  ijklcnt = (icut3 - ijklcnt);
  icut4 = icut3;
  if (icut3 > 0) 
    return ;
  printf("no two-electron integrals computed by node %d\n",GA_Nodeid());
  printf("\n");
}

int next_4chunk(int *lo,int *hi,int *ilo,int *jlo,int *klo,int *llo)
{
  int one = 0;
  long imax;
  long itask;
  int itmp;
  int ret;
  itask = NGA_Read_inc(g_counter,&one,1);
  imax = (nbfn / ichunk);
  if ((nbfn - (ichunk * imax)) > 0) 
    imax = (imax + 1);
  if (itask < 0) {
    printf("next_4chunk: itask negative: %d imax: %ld nbfn: %ld ichunk: %d\n",itask,imax,nbfn,ichunk);
    printf("probable GA int precision problem if imax^4 > 2^31\n");
    printf("\n");
    printf("next_4chunk\n");
    exit(0);
  }
  if (itask < (pow(imax,4))) {
     *ilo = (itask % imax);
    itmp = ((itask - ( *ilo)) / imax);
     *jlo = (itmp % imax);
    itmp = ((itmp -  *jlo) / imax);
     *klo = (itmp % imax);
     *llo = ((itmp -  *klo) / imax);
    lo[0] = ( *ilo * ichunk);
    lo[1] = ( *jlo * ichunk);
    lo[2] = ( *klo * ichunk);
    lo[3] = ( *llo * ichunk);
    hi[0] = MIN(((( *ilo + 1) * ichunk) - 1),nbfn);
    hi[1] = MIN(((( *jlo + 1) * ichunk) - 1),nbfn);
    hi[2] = MIN(((( *klo + 1) * ichunk) - 1),nbfn);
    hi[3] = MIN(((( *llo + 1) * ichunk) - 1),nbfn);
    ret = 1;
  }
  else 
    ret = 0;
  return ret;
}

long gettask()
{
  int offs = 0;
  long itask;
  itask = NGA_Read_inc(g_counter,&offs,1);
  return itask;
}

int translate_task(long itask,int *lo,int *hi,int *ilo,int *jlo,int *klo,int *llo)
{
  long imax;
  int itmp;
  int ret;
  imax = (nbfn / ichunk);
  if ((nbfn - (ichunk * imax)) > 0) 
    imax = (imax + 1);
  if (itask < 0) {
    printf("translate_task: itask negative: %ld imax: %ld nbfn: %ld ichunk: %d\n",itask,imax,nbfn,ichunk);
    printf("probable GA int precision problem if imax^4 > 2^31\n");
    printf("\n");
    printf("translate_task\n");
    exit(0);
  }
  if (itask < (pow(imax,4))) {
     *ilo = (itask % imax);
    itmp = ((itask - ( *ilo)) / imax);
     *jlo = (itmp % imax);
    itmp = ((itmp -  *jlo) / imax);
     *klo = (itmp % imax);
     *llo = ((itmp -  *klo) / imax);
    lo[0] = ( *ilo * ichunk);
    lo[1] = ( *jlo * ichunk);
    lo[2] = ( *klo * ichunk);
    lo[3] = ( *llo * ichunk);
    hi[0] = MIN(((( *ilo + 1) * ichunk) - 1),nbfn);
    hi[1] = MIN(((( *jlo + 1) * ichunk) - 1),nbfn);
    hi[2] = MIN(((( *klo + 1) * ichunk) - 1),nbfn);
    hi[3] = MIN(((( *llo + 1) * ichunk) - 1),nbfn);
    ret = 1;
  }
  else 
    ret = 0;
  return ret;
}
