
void sgemm_naive(int M,int N,int K,float alpha,const float *A,const float *B,float beta, float *C);
void sgemm_coalesce(int M,int N,int K,float alpha,const float *A,const float *B,float beta, float *C);