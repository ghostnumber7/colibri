/* Scale-sidecar geometry for the STREAMING expert path.
 *
 * The resident path (qt_from_disk) already upcasts BF16 .qs; the streaming path
 * -- which every routed expert takes -- assumed 4 bytes per scale in four
 * places. For Kimi-K2's gate_proj (O=2048, I=7168, BF16 .qs = 917504 bytes)
 * that resolves group size 64 instead of the true 32, with no error; it also
 * halves the fslab allocation and, in pin_arena_bind, under-allocates the
 * pinned arena 2x (a heap overrun, not just wrong numbers).
 *
 * fslab now always holds F32 scales; BF16 is read into the upper part of the
 * same buffer and upcast forward in place. This test pins the geometry, the
 * upcast, and the aliasing bound that makes the in-place walk safe.
 *
 * Round-2 review (I3): sections 1-4 pinned the geometry MATH on hand-built
 * QScales structs but never called qscales_plan/qscales_raw themselves, so the
 * code that actually implements the math was unpinned. Sections 5-7 below drive
 * qscales_plan with real st_tensor triples (BF16, F32, F16 -- M1/M2 accepts F16
 * now instead of refusing it) and check its dtype gate (mixed dtypes, bad byte
 * counts, unrecognized dtype, the NS bound); section 8 is the no-gap/no-overlap
 * invariant tying qscales_raw + qscales_upcast's landing zone to
 * qscales_alloc_bytes exactly, plus the monotone-in-NS regression check for I1
 * (required bytes must depend on NS alone, not on sbytes -- see the comment on
 * qscales_alloc_bytes in colibri.c). */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

static uint16_t f32_to_bf16(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }
static float bf16_trunc(float f){ uint32_t u; memcpy(&u,&f,4); u&=0xFFFF0000u; float r; memcpy(&r,&u,4); return r; }

/* Minimal, exact F16 encoder for round-trip testing (values chosen below are
 * exactly representable in F16, so this doesn't need to handle rounding). */
static uint16_t f32_to_f16_exact(float f){
    uint32_t u; memcpy(&u,&f,4);
    uint32_t sign=(u>>16)&0x8000, exp=(u>>23)&0xFF, man=u&0x7FFFFF;
    if(exp==0 && man==0) return (uint16_t)sign;                 /* +-0 */
    int e = (int)exp - 127 + 15;
    return (uint16_t)(sign | (uint32_t)(e<<10) | (man>>13));
}

/* Builds one st_tensor with the given dtype/nbytes; name/fd/off/numel are
 * unused by qscales_plan (it only reads ->dtype and ->nbytes). */
static st_tensor mk_tensor(int dtype, int64_t nbytes){
    st_tensor t; memset(&t,0,sizeof t);
    t.dtype=dtype; t.nbytes=nbytes; t.fd=-1;
    return t;
}

int main(void){
    /* ---- 1. K2's real gate_proj shape resolves gs=32, not gs=64 ---- */
    {
        const int O=2048, I=7168, GS=32;
        int64_t nb = (int64_t)O*((I+1)/2);              /* int4-packed weights */
        int64_t nscales = (int64_t)O*((I+GS-1)/GS);     /* 2048*224 = 458752 */
        int64_t ns_bf16 = nscales*2;                    /* 917504 bytes on disk */

        /* Wrong (what the streaming path did): raw bytes fed to qt_resolve_fmt */
        int gs_wrong=0;
        CHECK(qt_resolve_fmt("gate", O, I, nb, ns_bf16, &gs_wrong) == 4);
        CHECK(gs_wrong == 64);                          /* the silent bug, pinned */

        /* Right: F32-equivalent byte count */
        int gs_right=0;
        CHECK(qt_resolve_fmt("gate", O, I, nb, nscales*4, &gs_right) == 4);
        CHECK(gs_right == GS);
    }

    /* ---- 2. qscales_alloc_bytes / raw_off geometry, incl. the I1 fix:
     * required bytes are NOW a function of NS alone -- a BF16 and an F32
     * QScales with the SAME NS must get the SAME allocation. Before I1 they
     * didn't (F32 had no pad), which is exactly what let a slot last sized by
     * an F32 expert be silently reused (equal NS, so the "ftot > fslab_cap"
     * check didn't fire) by a BF16 expert whose upcast then ran 64 bytes past
     * the buffer's end. ---- */
    {
        QScales q; memset(&q,0,sizeof q);
        q.sbytes = 2; q.nsc[0]=100; q.nsc[1]=200; q.nsc[2]=300;
        q.NS = 600; q.raw_off = q.NS*2 + 64;
        CHECK(qscales_alloc_bytes(&q) == (size_t)(600*4 + 64));

        /* the aliasing bound: the LAST write must not reach the LAST source */
        int64_t last_write_end = 4*(q.NS-1) + 4;        /* 2400 */
        int64_t last_read_start = q.raw_off + 2*(q.NS-1); /* 1264+1198 = 2462 */
        CHECK(last_write_end <= last_read_start);

        QScales f; memset(&f,0,sizeof f);
        f.sbytes = 4; f.nsc[0]=100; f.nsc[1]=200; f.nsc[2]=300; f.NS=600; f.raw_off=0;
        /* I1: F32 now ALSO carries the pad -- monotone in NS alone. */
        CHECK(qscales_alloc_bytes(&f) == (size_t)(600*4 + 64));
        CHECK(qscales_alloc_bytes(&f) == qscales_alloc_bytes(&q));   /* same NS -> same size, any dtype */
    }

    /* ---- 3. in-place upcast is exact, for every element ---- */
    {
        const int64_t NS = 4096;                        /* spans the aliasing region */
        QScales q; memset(&q,0,sizeof q);
        q.sbytes=2; q.dt=0; q.nsc[0]=1024; q.nsc[1]=1024; q.nsc[2]=2048; q.NS=NS; q.raw_off=NS*2+64;

        float *fslab = malloc(qscales_alloc_bytes(&q));
        float *want  = malloc((size_t)NS*sizeof(float));
        if(!fslab || !want){ printf("FAIL %s:%d: OOM in test setup\n",__FILE__,__LINE__); fails++; return 1; }

        uint16_t *raw = (uint16_t*)((char*)fslab + q.raw_off);
        for(int64_t i=0;i<NS;i++){
            float v = (float)((i%977) - 488) * 1.0009765625f;   /* spread of magnitudes */
            want[i] = bf16_trunc(v);
            raw[i]  = f32_to_bf16(v);
        }
        qscales_upcast(fslab, &q);
        for(int64_t i=0;i<NS;i++) if(fslab[i] != want[i]){
            printf("FAIL upcast[%" PRId64 "]: %.9g want %.9g\n",i,(double)fslab[i],(double)want[i]); fails++; break;
        }
        free(fslab); free(want);
    }

    /* ---- 4. F32 path is a no-op (GLM safety) ---- */
    {
        const int64_t NS = 256;
        QScales q; memset(&q,0,sizeof q);
        q.sbytes=4; q.nsc[0]=64; q.nsc[1]=64; q.nsc[2]=128; q.NS=NS; q.raw_off=0;
        float *fslab = malloc(qscales_alloc_bytes(&q));
        float *copy  = malloc((size_t)NS*sizeof(float));
        if(!fslab || !copy){ printf("FAIL %s:%d: OOM in test setup\n",__FILE__,__LINE__); fails++; return 1; }
        for(int64_t i=0;i<NS;i++) fslab[i] = (float)i * 0.25f;
        memcpy(copy, fslab, (size_t)NS*sizeof(float));
        qscales_upcast(fslab, &q);
        CHECK(memcmp(fslab, copy, (size_t)NS*sizeof(float)) == 0);
        free(fslab); free(copy);
    }

    /* ---- 5. qscales_plan on REAL st_tensor triples: BF16 ---- */
    {
        const int GS=32, O=2048, I=7168;
        int64_t nscales = (int64_t)O*((I+GS-1)/GS);          /* 458752, matches section 1 */
        st_tensor tq_[3] = { mk_tensor(0, nscales*2), mk_tensor(0, nscales*2), mk_tensor(0, nscales*2) };
        st_tensor *tq[3] = { &tq_[0], &tq_[1], &tq_[2] };
        QScales q; int rc = qscales_plan(NULL, tq, &q);
        CHECK(rc==0);
        CHECK(q.sbytes==2); CHECK(q.dt==0);
        CHECK(q.nsc[0]==nscales); CHECK(q.nsc[1]==nscales); CHECK(q.nsc[2]==nscales);
        CHECK(q.NS==nscales*3);
        CHECK(q.raw_off == q.NS*2 + 64);
    }

    /* ---- 6. qscales_plan on REAL st_tensor triples: F32 (GLM) ---- */
    {
        const int O=2048;
        st_tensor tq_[3] = { mk_tensor(2,(int64_t)O*4), mk_tensor(2,(int64_t)O*4), mk_tensor(2,(int64_t)(O*2)*4) };
        st_tensor *tq[3] = { &tq_[0], &tq_[1], &tq_[2] };
        QScales q; int rc = qscales_plan(NULL, tq, &q);
        CHECK(rc==0);
        CHECK(q.sbytes==4); CHECK(q.dt==2);
        CHECK(q.nsc[0]==O); CHECK(q.nsc[1]==O); CHECK(q.nsc[2]==O*2);
        CHECK(q.NS==O*4);
        CHECK(q.raw_off==0);                       /* F32: no landing zone */

        /* the F32 branch returns plain float-scaled offsets, not the BF16
         * landing-zone formula -- qscales_raw(f,&q,k) must equal fslab+cumulative
         * scale count for every k, with NO dependence on raw_off/QSCALES_PAD. */
        float *fslab = malloc(qscales_alloc_bytes(&q));
        CHECK(fslab != NULL);
        if(fslab){
            CHECK(qscales_raw(fslab,&q,0) == (char*)(fslab+0));
            CHECK(qscales_raw(fslab,&q,1) == (char*)(fslab+q.nsc[0]));
            CHECK(qscales_raw(fslab,&q,2) == (char*)(fslab+q.nsc[0]+q.nsc[1]));
            free(fslab);
        }
    }

    /* ---- 7. qscales_plan on REAL st_tensor triples: F16 (M1/M2: decode, don't
     * refuse -- f16_to_f32 already exists and the resident path already uses it,
     * so refusing on the streaming path was an asymmetry, not a safety win) ---- */
    {
        const int64_t NS_EACH = 37;   /* small, odd count -- exercises the tail */
        st_tensor tq_[3] = { mk_tensor(1,NS_EACH*2), mk_tensor(1,NS_EACH*2), mk_tensor(1,NS_EACH*2) };
        st_tensor *tq[3] = { &tq_[0], &tq_[1], &tq_[2] };
        QScales q; int rc = qscales_plan(NULL, tq, &q);
        CHECK(rc==0);
        CHECK(q.sbytes==2); CHECK(q.dt==1);
        CHECK(q.NS==NS_EACH*3);

        float *fslab = malloc(qscales_alloc_bytes(&q));
        float *want  = malloc((size_t)q.NS*sizeof(float));
        if(!fslab || !want){ printf("FAIL %s:%d: OOM in test setup\n",__FILE__,__LINE__); fails++; return 1; }
        uint16_t *raw = (uint16_t*)((char*)fslab + q.raw_off);
        for(int64_t i=0;i<q.NS;i++){
            float v = (float)(i - q.NS/2) * 0.5f;   /* exactly F16-representable */
            want[i] = v;
            raw[i]  = f32_to_f16_exact(v);
        }
        qscales_upcast(fslab, &q);
        int ok=1;
        for(int64_t i=0;i<q.NS;i++) if(fslab[i] != want[i]){
            printf("FAIL f16 upcast[%" PRId64 "]: %.9g want %.9g\n",i,(double)fslab[i],(double)want[i]);
            fails++; ok=0; break;
        }
        (void)ok;
        free(fslab); free(want);
    }

    /* ---- 8. qscales_plan's dtype/size gate ---- */
    {
        /* mixed dtypes across gate/up/down: refuse */
        st_tensor mix_[3] = { mk_tensor(0,64), mk_tensor(2,128), mk_tensor(0,64) };
        st_tensor *mix[3] = { &mix_[0], &mix_[1], &mix_[2] };
        QScales q;
        CHECK(qscales_plan(NULL, mix, &q) == -1);

        /* nbytes not divisible by sbytes (BF16 = 2 bytes/scale, odd byte count) */
        st_tensor odd_[3] = { mk_tensor(0,65), mk_tensor(0,64), mk_tensor(0,64) };
        st_tensor *odd[3] = { &odd_[0], &odd_[1], &odd_[2] };
        CHECK(qscales_plan(NULL, odd, &q) == -1);

        /* zero/negative nbytes: refuse */
        st_tensor zero_[3] = { mk_tensor(0,0), mk_tensor(0,64), mk_tensor(0,64) };
        st_tensor *zero[3] = { &zero_[0], &zero_[1], &zero_[2] };
        CHECK(qscales_plan(NULL, zero, &q) == -1);

        /* unrecognized dtype (3 = U8, a real code in this codebase, just not a
         * valid .qs dtype): refuse, not a silent fallback to some default. */
        st_tensor bad_[3] = { mk_tensor(3,64), mk_tensor(3,64), mk_tensor(3,64) };
        st_tensor *bad[3] = { &bad_[0], &bad_[1], &bad_[2] };
        CHECK(qscales_plan(NULL, bad, &q) == -1);

        /* NS bound (M3, tightened again by NEW-4 to (SIZE_MAX-PAD)/4 so the pad
         * itself can't push qscales_alloc_bytes past SIZE_MAX): three tensors whose
         * SUMMED scale count exceeds the bound must be refused. Split across three
         * moderate tensors, not one -- a single tensor pushed anywhere near that
         * bound would need nbytes close to INT64_MAX, and computing that boundary
         * value in the test would itself risk the exact signed-overflow UB this
         * bound exists to prevent in the real code. */
        int64_t over_bound = (int64_t)((SIZE_MAX-QSCALES_PAD)/4) + 3;  /* just past the limit */
        int64_t nsc_each = over_bound/3 + 1;                 /* 3 tensors, sum > bound */
        int64_t nbytes_each = nsc_each*2;                    /* BF16: 2 bytes/scale */
        st_tensor huge_[3] = { mk_tensor(0,nbytes_each), mk_tensor(0,nbytes_each), mk_tensor(0,nbytes_each) };
        st_tensor *huge[3] = { &huge_[0], &huge_[1], &huge_[2] };
        CHECK(qscales_plan(NULL, huge, &q) == -1);

        /* Pin the TIGHTENING specifically (M3 raised the bound from 1<<40 to
         * ~SIZE_MAX/4): a value strictly between the two -- the OLD bound would
         * reject it, only the NEW one accepts it -- proves the bound actually
         * moved, not just that "some bound" rejects something enormous (both
         * 1<<40 and SIZE_MAX/4 reject `over_bound` above equally). */
        int64_t mid = (int64_t)1<<48;                        /* >> 1<<40, << (SIZE_MAX-PAD)/4 */
        int64_t nsc_mid_each = mid/3 + 1;
        int64_t nbytes_mid_each = nsc_mid_each*2;
        st_tensor mid_[3] = { mk_tensor(0,nbytes_mid_each), mk_tensor(0,nbytes_mid_each), mk_tensor(0,nbytes_mid_each) };
        st_tensor *midt[3] = { &mid_[0], &mid_[1], &mid_[2] };
        QScales qmid;
        CHECK(qscales_plan(NULL, midt, &qmid) == 0);
        CHECK(qmid.NS > ((int64_t)1<<40));
    }

    /* ---- 9. no-gap/no-overlap invariant: the raw landing zone's last tensor
     * ends EXACTLY at the allocation's end for a narrow (BF16/F16) sidecar --
     * there is no slack beyond QSCALES_PAD and no overrun. For F32 the raw
     * region (the final resting place itself) ends at NS*4, i.e. exactly
     * QSCALES_PAD bytes SHORT of qscales_alloc_bytes -- that trailing pad is
     * unconditional (I1) but unused/untouched by the F32 no-op path. ---- */
    {
        QScales q; memset(&q,0,sizeof q);
        q.sbytes=2; q.dt=0; q.nsc[0]=11; q.nsc[1]=13; q.nsc[2]=17; q.NS=41; q.raw_off=q.NS*2+64;
        /* Real allocations for the pointer arithmetic below (never dereferenced
         * past what qscales_alloc_bytes itself sized, so this stays UBSan-clean:
         * forming/advancing a pointer derived from an arbitrary integer, rather
         * than a genuine allocation, is its own flavor of undefined behavior even
         * when nothing is ever read through it). */
        float *fslab = malloc(qscales_alloc_bytes(&q));
        CHECK(fslab != NULL);
        if(fslab){
            char *end_of_raw = qscales_raw(fslab,&q,2) + q.nsc[2]*q.sbytes;
            char *end_of_alloc = (char*)fslab + qscales_alloc_bytes(&q);
            CHECK(end_of_raw == end_of_alloc);
            free(fslab);
        }

        QScales f; memset(&f,0,sizeof f);
        f.sbytes=4; f.nsc[0]=11; f.nsc[1]=13; f.nsc[2]=17; f.NS=41; f.raw_off=0;
        float *fslab_f = malloc(qscales_alloc_bytes(&f));
        CHECK(fslab_f != NULL);
        if(fslab_f){
            char *end_of_raw_f32 = qscales_raw(fslab_f,&f,2) + f.nsc[2]*f.sbytes;
            char *end_of_float_region = (char*)fslab_f + (size_t)f.NS*4;
            char *end_of_alloc_f32 = (char*)fslab_f + qscales_alloc_bytes(&f);
            CHECK(end_of_raw_f32 == end_of_float_region);          /* F32 data ends here... */
            CHECK(end_of_alloc_f32 == end_of_float_region + QSCALES_PAD);  /* ...QSCALES_PAD before alloc end */
            free(fslab_f);
        }
    }

    if(fails){ printf("qscales tests: %d FAILED\n", fails); return 1; }
    printf("qscales tests: ok (gs=32 not 64, geometry incl. I1 monotone-in-NS, "
           "in-place upcast exact BF16+F16, F32 no-op, qscales_plan dtype gate, "
           "no-gap/no-overlap invariant)\n");
    return 0;
}
