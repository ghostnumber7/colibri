/* Loader-seam test for BF16-stored quantization scales (.qs).
 *
 * Kimi-K2 ships its group scales as BF16 instead of F32 to halve the .qs
 * sidecar's bytes (and, downstream, the per-expert streaming bandwidth).
 * qt_from_disk previously assumed every .qs was F32 (4 bytes/scale) when it
 * computed the scale COUNT fed to qt_resolve_fmt (and, for fmt=4, to
 * detect_group_size) -- a BF16 sidecar has half the bytes of an F32 one for
 * the same scale count, so the derived group size / format was wrong.
 *
 * This test writes ONE grouped-int4 (fmt=4) tensor and ONE plain per-row
 * int4 tensor (fmt=2), each with TWO .qs sidecars holding the SAME scale
 * values -- one stored F32, one stored BF16 (values pre-truncated to BF16
 * precision so the round trip is bit-exact, not just close) -- and checks
 * that qt_from_disk on the BF16 sidecar:
 *   - derives the same fmt/gs as the F32 sidecar,
 *   - upcasts to BIT-IDENTICAL float scales, and
 *   - produces a bit-identical matmul_i4_grouped result,
 * proving BF16 .qs support without touching the F32 path (GLM safety: the
 * F32 branch takes sbytes=4 -> ns4==ns, i.e. the exact same code as before). */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

static uint64_t rng = 0x1234ABCD9876FEDCull;
static float rndf(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return ((int64_t)(rng & 0xFFFFF) - 0x80000) / (float)0x80000; }

/* Truncate a float to BF16 precision (zero the low 16 mantissa bits) so that
 * writing it as F32 and writing it as BF16 (top 16 bits) round-trip to the
 * SAME float -- lets the test assert bit-exact equality instead of "close". */
static float bf16_trunc(float f){
    uint32_t u; memcpy(&u,&f,4); u &= 0xFFFF0000u;
    float r; memcpy(&r,&u,4); return r;
}
static uint16_t f32_to_bf16(float f){
    uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16);
}

/* Simple grouped-int4 packer (offset encoding, nib-8, matching quant.h's
 * matmul_i4_grouped / test_i4_grouped.c's reference): per-row, per-group
 * scale = max(|w|)/7, clamped nibble. */
static void pack_i4_grouped(const float *w, uint8_t *q4, float *scale, int O, int I, int gs){
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    memset(q4,0,(size_t)O*rb);
    for(int o=0;o<O;o++){
        const float *wr = w + (int64_t)o*I;
        float *srow = scale + (int64_t)o*ng;
        for(int g=0; g<ng; g++){
            int i0=g*gs, i1=i0+gs; if(i1>I) i1=I;
            float maxabs=0;
            for(int i=i0;i<i1;i++){ float a=fabsf(wr[i]); if(a>maxabs) maxabs=a; }
            /* pre-truncate to BF16 precision: the F32 and BF16 sidecars then
             * carry the identical value, and the upcast is bit-exact. */
            srow[g]=bf16_trunc(maxabs>0 ? maxabs/7.0f : 1.0f);
        }
        uint8_t *row = q4 + (int64_t)o*rb;
        for(int i=0;i<I;i++){
            float sc=srow[i/gs];
            int nib=(int)lroundf(wr[i]/sc)+8;
            if(nib<0) nib=0; if(nib>15) nib=15;
            if(i&1) row[i>>1] |= (uint8_t)(nib<<4);
            else    row[i>>1] |= (uint8_t)nib;
        }
    }
}

/* Plain per-row int4 (fmt=2 on disk): one scale per row. */
static void pack_i4_row(const float *w, uint8_t *q4, float *scale, int O, int I){
    int rb=(I+1)/2;
    memset(q4,0,(size_t)O*rb);
    for(int o=0;o<O;o++){
        const float *wr=w+(int64_t)o*I;
        float maxabs=0; for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>maxabs) maxabs=a; }
        float sc = bf16_trunc(maxabs>0 ? maxabs/7.0f : 1.0f);
        scale[o]=sc;
        uint8_t *row=q4+(int64_t)o*rb;
        for(int i=0;i<I;i++){
            int nib=(int)lroundf(wr[i]/sc)+8;
            if(nib<0) nib=0; if(nib>15) nib=15;
            if(i&1) row[i>>1] |= (uint8_t)(nib<<4);
            else    row[i>>1] |= (uint8_t)nib;
        }
    }
}

static void append_bf16(FILE *f, const float *v, int64_t n){
    for(int64_t i=0;i<n;i++){ uint16_t h=f32_to_bf16(v[i]); fwrite(&h,2,1,f); }
}

int main(void){
    enum { O=6, I=320, GS=64 };                 /* 320/64 = 5 groups/row, matches
                                                  * the g64 shape convention used elsewhere */
    int64_t ng=(I+GS-1)/GS, rb=(I+1)/2;

    static float w[O*I];
    for(int i=0;i<O*I;i++) w[i]=rndf()*0.05f;

    static uint8_t q4g[O*((I+1)/2)]; static float sg[O*5];   /* grouped, fmt=4 */
    pack_i4_grouped(w,q4g,sg,O,I,GS);
    static uint8_t q4r[O*((I+1)/2)]; static float sr[O];     /* per-row, fmt=2 */
    pack_i4_row(w,q4r,sr,O,I);

    const char *dir="tests/tmp_bf16_scales_snap";
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir, 0755);
#endif
    char path[256]; snprintf(path,sizeof path,"%s/model.safetensors",dir);

    int64_t nb_g=(int64_t)O*rb, ns_g_f32=(int64_t)O*ng*4, ns_g_bf16=(int64_t)O*ng*2;
    int64_t nb_r=(int64_t)O*rb, ns_r_f32=(int64_t)O*4,     ns_r_bf16=(int64_t)O*2;

    /* Layout: wg (weights, shared by both grouped sidecars) | wg.qs (F32) |
     * wg_bf (weights, shared by both grouped sidecars) | wg_bf.qs (BF16) |
     * wr (row weights) | wr.qs (F32) | wr_bf (row weights) | wr_bf.qs (BF16) */
    int64_t off=0;
    int64_t o_wg=off; off+=nb_g;
    int64_t o_wg_qs=off; off+=ns_g_f32;
    int64_t o_wgb=off; off+=nb_g;
    int64_t o_wgb_qs=off; off+=ns_g_bf16;
    int64_t o_wr=off; off+=nb_r;
    int64_t o_wr_qs=off; off+=ns_r_f32;
    int64_t o_wrb=off; off+=nb_r;
    int64_t o_wrb_qs=off; off+=ns_r_bf16;

    char hdr[2048];
    int hl=snprintf(hdr,sizeof hdr,
        "{\"wg\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"wg.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"wgb\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"wgb.qs\":{\"dtype\":\"BF16\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"wr\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"wr.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"wrb\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"wrb.qs\":{\"dtype\":\"BF16\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}}",
        (long long)nb_g,(long long)o_wg,(long long)(o_wg+nb_g),
        (long long)(O*ng),(long long)o_wg_qs,(long long)(o_wg_qs+ns_g_f32),
        (long long)nb_g,(long long)o_wgb,(long long)(o_wgb+nb_g),
        (long long)(O*ng),(long long)o_wgb_qs,(long long)(o_wgb_qs+ns_g_bf16),
        (long long)nb_r,(long long)o_wr,(long long)(o_wr+nb_r),
        (long long)O,(long long)o_wr_qs,(long long)(o_wr_qs+ns_r_f32),
        (long long)nb_r,(long long)o_wrb,(long long)(o_wrb+nb_r),
        (long long)O,(long long)o_wrb_qs,(long long)(o_wrb_qs+ns_r_bf16));
    if(hl<0 || hl>=(int)sizeof hdr){ printf("FAIL: header too small\n"); return 1; }

    FILE *f=fopen(path,"wb");
    if(!f){ printf("FAIL: cannot create %s (run from c/, like tools/run_tests.py does)\n", path); return 1; }
    uint64_t hlen=(uint64_t)hl;
    fwrite(&hlen,8,1,f); fwrite(hdr,1,hl,f);
    fwrite(q4g,1,(size_t)nb_g,f); fwrite(sg,1,(size_t)ns_g_f32,f);
    fwrite(q4g,1,(size_t)nb_g,f); append_bf16(f,sg,O*ng);
    fwrite(q4r,1,(size_t)nb_r,f); fwrite(sr,1,(size_t)ns_r_f32,f);
    fwrite(q4r,1,(size_t)nb_r,f); append_bf16(f,sr,O);
    fclose(f);

    static Model gm;                            /* only gm.S is used by qt_from_disk */
    st_init(&gm.S, dir);

    /* dtype accessor sanity: F32/BF16 sidecars report the dtype we wrote */
    CHECK(st_dtype(&gm.S,"wg.qs")==2);
    CHECK(st_dtype(&gm.S,"wgb.qs")==0);
    CHECK(st_dtype(&gm.S,"missing.name")==-1);

    /* ---- grouped int4 (fmt=4): F32 vs BF16 .qs must agree exactly ---- */
    QT tg32; memset(&tg32,0,sizeof tg32);
    qt_from_disk(&gm,"wg",O,I,4,0,&tg32);
    CHECK(tg32.fmt==4); CHECK(tg32.gs==GS);

    QT tgbf; memset(&tgbf,0,sizeof tgbf);
    qt_from_disk(&gm,"wgb",O,I,4,0,&tgbf);
    CHECK(tgbf.fmt==4); CHECK(tgbf.gs==GS);

    CHECK(memcmp(tg32.s,tgbf.s,(size_t)O*ng*sizeof(float))==0);   /* bit-identical scales */
    CHECK(memcmp(tg32.q4,tgbf.q4,(size_t)nb_g)==0);               /* weights untouched either way */

    static float x[I], y32[O], ybf[O];
    for(int i=0;i<I;i++) x[i]=rndf();
    /* s_bf16 comes from the QT itself: qt_from_disk is the RESIDENT path, which always
     * widens to f32, so both are 0 here -- and stay correct if that ever changes. */
    matmul_i4_grouped(y32,x,tg32.q4,tg32.s,1,I,O,tg32.gs,tg32.s_bf16);
    matmul_i4_grouped(ybf,x,tgbf.q4,tgbf.s,1,I,O,tgbf.gs,tgbf.s_bf16);
    CHECK(memcmp(y32,ybf,sizeof y32)==0);                         /* bit-identical matmul output */

    /* ---- plain per-row int4 (fmt=2): BF16 .qs also resolves correctly ---- */
    QT tr32; memset(&tr32,0,sizeof tr32);
    qt_from_disk(&gm,"wr",O,I,4,0,&tr32);
    CHECK(tr32.fmt==2);

    QT trbf; memset(&trbf,0,sizeof trbf);
    qt_from_disk(&gm,"wrb",O,I,4,0,&trbf);
    CHECK(trbf.fmt==2);

    CHECK(memcmp(tr32.s,trbf.s,(size_t)O*sizeof(float))==0);
    CHECK(memcmp(tr32.q4,trbf.q4,(size_t)nb_r)==0);

    unlink(path); rmdir(dir);
    if(fails){ printf("bf16 scales loader tests: %d FAILED\n", fails); return 1; }
    printf("bf16 scales loader tests: ok (fmt=4 gs=%d, fmt=2, F32 and BF16 .qs bit-identical)\n", GS);
    return 0;
}
