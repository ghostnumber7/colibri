/* fmt=7 (mxfp4, Kimi-K3 experts) — kernel oracle + loader geometry.
 *
 * The mxfp4 container is a RAW PASSTHROUGH of the compressed-tensors source
 * (e2m1 nibbles + U8 e8m0 group-32 scales, transcode_mxfp4 in the converter),
 * so the ONLY place the decode contract is implemented engine-side is
 * matmul_mxfp4. Its nibble bytes are byte-identical to fmt=2/4 int4 — the
 * exact silent-mis-decode hazard this suite pins:
 *
 * 1) kernel vs an INDEPENDENT scalar reference (own LUT, double accumulate)
 *    on shapes that exercise the AVX2 body and the scalar tail;
 * 2) qt_resolve_fmt routes U8-sidecar tensors to fmt=7/gs=32 and NEVER lets
 *    byte counts alone route them into the int4 arms;
 * 3) qscales_plan accepts a U8 sidecar with sbytes=1, lands it RAW at the
 *    fslab base (raw_off=0), keep_raw is unconditional, and upcast is a no-op
 *    (an upcast would reinterpret exponent bytes as bf16 halves — garbage). */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

/* Independent reference: own LUT (typed out from the e2m1 spec, NOT shared with
 * quant.h), double accumulation, no grouping tricks. */
static const double REF_E2M1[16] = {0,0.5,1,1.5,2,3,4,6, -0.0,-0.5,-1,-1.5,-2,-3,-4,-6};
static void ref_mxfp4(float *y, float *mag, const float *x, const uint8_t *q4, const uint8_t *qs,
                      int S, int I, int O){
    int rb=(I+1)/2, ng=(I+31)/32;
    for(int o=0;o<O;o++) for(int s=0;s<S;s++){
        double a=0, m=0;
        for(int i=0;i<I;i++){
            uint8_t byte=q4[(int64_t)o*rb+(i>>1)];
            int nib=(i&1)?(byte>>4)&0xF:byte&0xF;
            double sc=ldexp(1.0,(int)qs[(int64_t)o*ng+i/32]-127);
            double t=(double)x[(int64_t)s*I+i]*REF_E2M1[nib]*sc;
            a += t; m += fabs(t);
        }
        y[(int64_t)s*O+o]=(float)a;
        mag[(int64_t)s*O+o]=(float)m;   /* sum of |terms|: the f32-accumulation error bound */
    }
}

static uint32_t rng_state=0x12345678u;
static uint32_t rnd(void){ rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5; return rng_state; }

static void test_kernel_shape(int S, int I, int O){
    int rb=(I+1)/2, ng=(I+31)/32;
    uint8_t *q4=malloc((size_t)O*rb), *qs=malloc((size_t)O*ng);
    float *x=malloc((size_t)S*I*sizeof(float));
    float *y=malloc((size_t)S*O*sizeof(float)), *yr=malloc((size_t)S*O*sizeof(float));
    float *ym=malloc((size_t)S*O*sizeof(float));
    for(int64_t i=0;i<(int64_t)O*rb;i++) q4[i]=(uint8_t)rnd();
    /* e8m0 spread around the real container's observed range (112..122 ~= 2^-15..2^-5),
     * plus a few extremes to catch a scale table indexed off by one */
    for(int64_t i=0;i<(int64_t)O*ng;i++) qs[i]=(uint8_t)(100+(rnd()%40));
    qs[0]=90; qs[(int64_t)O*ng-1]=140;
    for(int64_t i=0;i<(int64_t)S*I;i++) x[i]=((int)(rnd()%2000)-1000)/500.0f;
    matmul_mxfp4(y,x,q4,qs,S,I,O);
    ref_mxfp4(yr,ym,x,q4,qs,S,I,O);
    for(int64_t i=0;i<(int64_t)S*O;i++){
        float d=fabsf(y[i]-yr[i]);
        /* f32 accumulation error scales with the sum of |terms| (cancellation makes
         * a result-relative bound wrong), not with the result. */
        float tol=1e-6f+4e-6f*ym[i];
        if(d>tol){ printf("FAIL kernel S=%d I=%d O=%d idx=%lld: %g vs %g (mag %g)\n",
                          S,I,O,(long long)i,y[i],yr[i],ym[i]); fails++; break; }
    }
    free(q4); free(qs); free(x); free(y); free(yr); free(ym);
}

int main(void){
    /* 1) kernel oracle: AVX2 body (I%16==0), partial-tail group (I=32*3), S>1 */
    test_kernel_shape(1, 64, 7);
    test_kernel_shape(2, 96, 5);
    test_kernel_shape(3, 32, 4);
    test_kernel_shape(1, 3584, 8);    /* the real K3 expert input dim */

    /* 2) qt_resolve_fmt: U8 sidecar routes to fmt=7/gs=32. ns is passed in
     * F32-EQUIVALENT bytes (count*4), matching both callers. */
    { int gs=-1; int O=8, I=64; int64_t ngm=(I+31)/32;
      int fmt=qt_resolve_fmt("t.u8", O, I, (int64_t)O*((I+1)/2), (int64_t)O*ngm*4, 3, &gs);
      CHECK(fmt==7); CHECK(gs==32); }
    /* same byte counts with an F32 sidecar dtype must NOT become fmt=7: it is a
     * legitimate grouped-int4 (fmt=4, gs=32) container. */
    { int gs=-1; int O=8, I=64; int64_t ngm=(I+31)/32;
      int fmt=qt_resolve_fmt("t.f32", O, I, (int64_t)O*((I+1)/2), (int64_t)O*ngm*4, 2, &gs);
      CHECK(fmt==4); CHECK(gs==32); }

    /* 3) qscales_plan geometry for a U8 sidecar */
    { st_tensor a={0},b={0},c={0}; st_tensor *tq[3]={&a,&b,&c};
      a.dtype=b.dtype=c.dtype=3;             /* U8 */
      a.nbytes=8*2; b.nbytes=8*2; c.nbytes=16;   /* arbitrary counts, 1B/scale */
      QScales q;
      CHECK(qscales_plan(NULL,tq,&q)==0);
      CHECK(q.sbytes==1);
      CHECK(q.dt==3);
      CHECK(q.NS==16+16+16);
      CHECK(q.raw_off==0);                   /* raw u8 lands at the fslab base */
      CHECK(qscales_keep_raw(&q)==1);        /* unconditional for u8 -- no g_qs_bf16 gate */
      float *fslab=calloc(1,qscales_alloc_bytes(&q));
      CHECK((char*)qscales_raw(fslab,&q,0)==(char*)fslab);
      CHECK((char*)qscales_raw(fslab,&q,1)==(char*)fslab+16);
      CHECK((char*)qscales_raw(fslab,&q,2)==(char*)fslab+32);
      /* upcast must be a NO-OP: fill the raw zone, call it, bytes unchanged */
      memset(fslab,0xAB,48);
      qscales_upcast(fslab,&q);
      for(int i=0;i<48;i++) CHECK(((uint8_t*)fslab)[i]==0xAB);
      free(fslab); }

    /* mixed U8 + BF16 sidecars across gate/up/down must refuse */
    { st_tensor a={0},b={0},c={0}; st_tensor *tq[3]={&a,&b,&c};
      a.dtype=3; b.dtype=0; c.dtype=3;
      a.nbytes=b.nbytes=c.nbytes=16;
      QScales q;
      CHECK(qscales_plan(NULL,tq,&q)==-1); }

    /* 4) qt_bytes accounting for fmt=7 */
    { QT t; memset(&t,0,sizeof t); t.fmt=7; t.O=8; t.I=96; t.gs=32;
      CHECK(qt_bytes(&t)==(int64_t)8*48+8*3); }

    if(fails){ printf("%d FAILURES\n",fails); return 1; }
    printf("test_mxfp4_kernel: all OK\n");
    return 0;
}
