/* pin_arena_bind + expert_load_impl, ASan/UBSan regression for the 2x pinned-
 * arena under-allocation.
 *
 * pin_arena_bind sizes ONE per-layer arena from the layer's FIRST expert and
 * slices it at a fixed stride (`s->fslab = af + i*fs`) across every pinned
 * expert of that layer. Before this task, `fs` was derived from the .qs
 * sidecar's RAW byte count (`qtot/4*sizeof(float)`), which is correct for F32
 * scales but HALF of what a BF16 sidecar (Kimi-K2) actually needs once
 * qscales_upcast decodes it to F32 in place -- every slice after the first was
 * undersized by exactly the BF16/F32 byte-count ratio, so an expert's own
 * scale read+upcast ran past its own slice into the NEXT expert's, or past the
 * whole arena's end for the last slot. That is live heap corruption on any
 * K2-shaped container with >=2 pinned experts per layer, not just wrong
 * numbers.
 *
 * This container can't be a GLM (F32 .qs) fixture: for F32, qtot/4*sizeof(float)
 * was ALREADY correct (qtot/4 floats * 4 bytes/float == qtot bytes), so the
 * historical bug is BF16-specific and only reproducible with a BF16 sidecar.
 *
 * Uses THREE pinned experts (cnt[l]==3), not one or two: a single slot at
 * i=0 never exercises the `af + i*fs` stride at all (i*fs==0), and "this
 * expert's slice runs into its neighbour's" -- the actual damage mode, not a
 * generic OOB -- needs at least one interior slot with a neighbour on BOTH
 * sides. Experts are loaded LAST-index-first (2, 1, 0): qscales_upcast always
 * writes FORWARD within a slot (toward higher addresses), so if slot i's
 * write were to overrun into slot i+1's region, loading i+1 BEFORE i lets the
 * final bit-exact check on slot i+1's values (done only after ALL loads)
 * catch a corruption that a clean ASan run alone would miss -- only the very
 * LAST slot's overrun reaches unallocated memory; a middle slot's overrun
 * lands inside the arena's own allocation and silently clobbers a neighbour.
 *
 * Scale/shape sizing (NS=2000/expert: gate=up=800, down=400) is deliberately
 * NOT small: pin_arena_bind's slice size is rounded up to a 4096-byte page, so
 * a tiny NS lets page rounding absorb the entire old-vs-new byte-count
 * difference and this test would pass "clean" whether or not the bug were
 * present. At NS=2000, the old (buggy) byte count (2000*2=4000B) rounds to
 * ONE page; the correct one (2000*4+64=8064B) rounds to TWO -- a real,
 * ASan-visible page-level gap if the bug were reintroduced. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

static uint16_t f32_to_bf16(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }
static float bf16_trunc(float f){ uint32_t u; memcpy(&u,&f,4); u&=0xFFFF0000u; float r; memcpy(&r,&u,4); return r; }
static void append_bf16(FILE *f, const float *v, int64_t n){
    for(int64_t i=0;i<n;i++){ uint16_t h=f32_to_bf16(v[i]); fwrite(&h,2,1,f); }
}

enum { NE=3, I=800, D=400 };   /* moe_inter=I, hidden=D; NS/expert = I+I+D = 2000 */

static uint8_t s_wgate[NE][(size_t)I*D], s_wup[NE][(size_t)I*D], s_wdown[NE][(size_t)D*I];
static float   s_sgate[NE][I], s_sup[NE][I], s_sdown[NE][D];

/* ---- drive an arena slot into the realloc branch ------------------------------
 *
 * The scenario above never exercises expert_load_impl's slab/fslab realloc
 * guards at all (I5's own subject): all three experts share IDENTICAL shapes,
 * so nothing ever outgrows the arena's per-slot budget. This function builds a
 * SEPARATE 2-expert fixture where expert 0 (the "representative" pin_arena_bind
 * sizes the whole arena from) is small, and expert 1's down_proj is deliberately
 * re-encoded as GROUPED int4 (fmt=4, gs=16) instead of the plain per-row fmt=1
 * every other tensor uses -- same [O,I] shape, so expert_load_impl's own
 * per-tensor O/I derivation (from Cfg, shared across every expert) is
 * untouched, but grouped scales at gs=16 need O*ceil(I/16) values instead of
 * O, which is enough to blow past the arena's fslab_cap while the WEIGHT bytes
 * (int4-packed, actually smaller than fmt=1's) stay comfortably under the
 * arena's slab_cap. That is deliberate: it proves the slab and fslab realloc
 * guards fire INDEPENDENTLY -- expert 1's weight buffer stays arena-owned
 * (aslab survives) while its scale buffer detaches (afslab -> NULL) -- which is
 * exactly the asymmetric state the I5 review found `expert_host_release`/
 * `expert_host_ensure` mishandling when they tested `aslab` alone for both
 * pointers.
 *
 * Before I5's follow-up fix, this scenario would have `free()`'d an interior
 * arena pointer inside expert_load_impl's fslab-realloc block the moment
 * expert 1 loaded -- a `free(): invalid pointer` abort, or under ASan a
 * "attempting free on address which was not malloc()-ed" report. With the fix,
 * it must complete cleanly AND leave expert 0's still-arena-resident scale
 * values bit-exact (proving the detach-and-grow for slot 1 didn't corrupt its
 * neighbour's live slice of the same arena allocation). */
static int test_arena_realloc_guard(void){
    int local_fails = 0;
#define RCHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); local_fails++; } }while(0)

    const int GS = 16;                          /* smallest candidate detect_group_size tries */
    const int ng = (I + GS - 1) / GS;            /* groups per row at gs=16 */
    const int64_t down_ns_e1 = (int64_t)D * ng;  /* expert 1's down_proj scale count: >> fslab_cap */

    uint8_t *wgate0=calloc((size_t)I*D,1), *wup0=calloc((size_t)I*D,1), *wdown0=calloc((size_t)D*I,1);
    uint8_t *wgate1=calloc((size_t)I*D,1), *wup1=calloc((size_t)I*D,1);
    uint8_t *wdown1=calloc((size_t)D*((I+1)/2),1);         /* fmt=4: packed nibbles, half fmt=1's bytes */
    float *sgate0=malloc((size_t)I*sizeof(float)), *sup0=malloc((size_t)I*sizeof(float)), *sdown0=malloc((size_t)D*sizeof(float));
    float *sgate1=malloc((size_t)I*sizeof(float)), *sup1=malloc((size_t)I*sizeof(float)), *sdown1=malloc((size_t)down_ns_e1*sizeof(float));
    if(!wgate0||!wup0||!wdown0||!wgate1||!wup1||!wdown1||!sgate0||!sup0||!sdown0||!sgate1||!sup1||!sdown1){
        printf("FAIL %s:%d: OOM in test setup\n",__FILE__,__LINE__); return 1;
    }
    for(int i=0;i<I;i++){ sgate0[i]=bf16_trunc((float)(i)*0.001f+0.01f); sgate1[i]=bf16_trunc((float)(i)*0.001f+0.51f); }
    for(int i=0;i<I;i++){ sup0[i]  =bf16_trunc((float)(i)*0.001f+0.02f); sup1[i]  =bf16_trunc((float)(i)*0.001f+0.52f); }
    for(int i=0;i<D;i++)   sdown0[i]=bf16_trunc((float)(i)*0.001f+0.03f);
    for(int64_t i=0;i<down_ns_e1;i++) sdown1[i]=bf16_trunc((float)(i%997)*0.0001f+0.53f);

    const char *dir = "tests/tmp_pin_arena_realloc_snap";
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir, 0755);
#endif
    char path[300]; snprintf(path,sizeof path,"%s/model.safetensors",dir);

    int64_t off=0;
    int64_t o_wg0=off; off+=(int64_t)I*D;
    int64_t o_sg0=off; off+=(int64_t)I*2;
    int64_t o_wu0=off; off+=(int64_t)I*D;
    int64_t o_su0=off; off+=(int64_t)I*2;
    int64_t o_wd0=off; off+=(int64_t)D*I;
    int64_t o_sd0=off; off+=(int64_t)D*2;
    int64_t o_wg1=off; off+=(int64_t)I*D;
    int64_t o_sg1=off; off+=(int64_t)I*2;
    int64_t o_wu1=off; off+=(int64_t)I*D;
    int64_t o_su1=off; off+=(int64_t)I*2;
    int64_t o_wd1=off; off+=(int64_t)D*((I+1)/2);
    int64_t o_sd1=off; off+=down_ns_e1*2;

    char hdr[2048];
    int hl=snprintf(hdr,sizeof hdr,
        "{\"model.layers.0.mlp.experts.0.gate_proj.weight\":{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]},"
        "\"model.layers.0.mlp.experts.0.gate_proj.weight.qs\":{\"dtype\":\"BF16\",\"shape\":[%d],\"data_offsets\":[%lld,%lld]},"
        "\"model.layers.0.mlp.experts.0.up_proj.weight\":{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]},"
        "\"model.layers.0.mlp.experts.0.up_proj.weight.qs\":{\"dtype\":\"BF16\",\"shape\":[%d],\"data_offsets\":[%lld,%lld]},"
        "\"model.layers.0.mlp.experts.0.down_proj.weight\":{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]},"
        "\"model.layers.0.mlp.experts.0.down_proj.weight.qs\":{\"dtype\":\"BF16\",\"shape\":[%d],\"data_offsets\":[%lld,%lld]},"
        "\"model.layers.0.mlp.experts.1.gate_proj.weight\":{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]},"
        "\"model.layers.0.mlp.experts.1.gate_proj.weight.qs\":{\"dtype\":\"BF16\",\"shape\":[%d],\"data_offsets\":[%lld,%lld]},"
        "\"model.layers.0.mlp.experts.1.up_proj.weight\":{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]},"
        "\"model.layers.0.mlp.experts.1.up_proj.weight.qs\":{\"dtype\":\"BF16\",\"shape\":[%d],\"data_offsets\":[%lld,%lld]},"
        "\"model.layers.0.mlp.experts.1.down_proj.weight\":{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]},"
        "\"model.layers.0.mlp.experts.1.down_proj.weight.qs\":{\"dtype\":\"BF16\",\"shape\":[%d],\"data_offsets\":[%lld,%lld]}}",
        I,D,(long long)o_wg0,(long long)(o_wg0+(int64_t)I*D),
        I,(long long)o_sg0,(long long)(o_sg0+(int64_t)I*2),
        I,D,(long long)o_wu0,(long long)(o_wu0+(int64_t)I*D),
        I,(long long)o_su0,(long long)(o_su0+(int64_t)I*2),
        D,I,(long long)o_wd0,(long long)(o_wd0+(int64_t)D*I),
        D,(long long)o_sd0,(long long)(o_sd0+(int64_t)D*2),
        I,D,(long long)o_wg1,(long long)(o_wg1+(int64_t)I*D),
        I,(long long)o_sg1,(long long)(o_sg1+(int64_t)I*2),
        I,D,(long long)o_wu1,(long long)(o_wu1+(int64_t)I*D),
        I,(long long)o_su1,(long long)(o_su1+(int64_t)I*2),
        D,I,(long long)o_wd1,(long long)(o_wd1+(int64_t)D*((I+1)/2)),
        (int)down_ns_e1,(long long)o_sd1,(long long)(o_sd1+down_ns_e1*2));
    if(hl<0 || hl>=(int)sizeof hdr){ printf("FAIL: header too small\n"); return 1; }

    FILE *f=fopen(path,"wb");
    if(!f){ printf("FAIL: cannot create %s\n", path); return 1; }
    uint64_t hlen=(uint64_t)hl;
    fwrite(&hlen,8,1,f); fwrite(hdr,1,hl,f);
    fwrite(wgate0,1,(size_t)I*D,f); append_bf16(f,sgate0,I);
    fwrite(wup0,1,(size_t)I*D,f);   append_bf16(f,sup0,I);
    fwrite(wdown0,1,(size_t)D*I,f); append_bf16(f,sdown0,D);
    fwrite(wgate1,1,(size_t)I*D,f); append_bf16(f,sgate1,I);
    fwrite(wup1,1,(size_t)I*D,f);   append_bf16(f,sup1,I);
    fwrite(wdown1,1,(size_t)D*((I+1)/2),f); append_bf16(f,sdown1,down_ns_e1);
    fclose(f);

    static Model m2; memset(&m2,0,sizeof m2);
    st_init(&m2.S, dir);
    m2.c.n_layers=1; m2.c.hidden=D; m2.c.moe_inter=I; m2.ebits=8;
    g_numa_nodes = 2;

    int NR = m2.c.n_layers+1;
    m2.pin  = calloc((size_t)NR, sizeof(ESlot*));
    m2.npin = calloc((size_t)NR, sizeof(int));

    PinRec r2[2] = { {0,0,999}, {0,1,888} };
    int slot_of2[2] = {0,1};
    m2.npin[0]=2;
    m2.pin[0]=calloc(2, sizeof(ESlot));

    pin_arena_bind(&m2, r2, slot_of2, 0, 2);
    RCHECK(m2.pin[0][0].aslab && m2.pin[0][0].afslab);
    RCHECK(m2.pin[0][1].aslab && m2.pin[0][1].afslab);   /* both start fully arena-bound */

    uint8_t *slab0_before=m2.pin[0][0].slab, *slab1_before=m2.pin[0][1].slab;
    float   *fslab0_before=m2.pin[0][0].fslab, *fslab1_before=m2.pin[0][1].fslab;

    int rc0 = expert_load_impl(&m2, 0, 0, &m2.pin[0][slot_of2[0]], 1, 0);
    RCHECK(rc0==0);
    RCHECK(m2.pin[0][0].slab==slab0_before && m2.pin[0][0].fslab==fslab0_before);   /* fits: no realloc at all */
    RCHECK(m2.pin[0][0].aslab && m2.pin[0][0].afslab);

    int rc1 = expert_load_impl(&m2, 0, 1, &m2.pin[0][slot_of2[1]], 1, 0);
    RCHECK(rc1==0);
    /* THE point of this test: the two guards fire INDEPENDENTLY. Weight bytes
     * (int4-packed) are smaller than expert 0's fmt=1 weights, so slab still
     * fits -> stays arena-owned, unchanged pointer. Scale count (gs=16 grouped)
     * is >>fslab_cap -> fslab detaches and gets a fresh individual allocation. */
    RCHECK(m2.pin[0][1].aslab != NULL);
    RCHECK(m2.pin[0][1].slab == slab1_before);
    RCHECK(m2.pin[0][1].afslab == NULL);
    RCHECK(m2.pin[0][1].fslab != fslab1_before);
    RCHECK(m2.pin[0][1].fslab != NULL);
    RCHECK(m2.pin[0][1].fslab_cap == down_ns_e1 + I + I);   /* == this expert's own qs.NS, not the arena's */

    /* Slot 0's arena-resident scales must be UNCHANGED by slot 1's detach+grow
     * (this is what an interior-pointer free() or an undersized neighbour slice
     * would corrupt, silently, with no ASan signal of its own -- the value
     * check is the only thing that catches it). */
    {
        QT *qt0[3] = { &m2.pin[0][0].g, &m2.pin[0][0].u, &m2.pin[0][0].d };
        for(int i=0;i<I;i++) RCHECK(qt0[0]->s[i]==sgate0[i]);
        for(int i=0;i<I;i++) RCHECK(qt0[1]->s[i]==sup0[i]);
        for(int i=0;i<D;i++) RCHECK(qt0[2]->s[i]==sdown0[i]);
    }
    /* Slot 1's own (freshly, individually allocated) scales must be correct too. */
    {
        QT *qt1[3] = { &m2.pin[0][1].g, &m2.pin[0][1].u, &m2.pin[0][1].d };
        for(int i=0;i<I;i++) RCHECK(qt1[0]->s[i]==sgate1[i]);
        for(int i=0;i<I;i++) RCHECK(qt1[1]->s[i]==sup1[i]);
        for(int64_t i=0;i<down_ns_e1;i++) RCHECK(qt1[2]->s[i]==sdown1[i]);
    }

    free(wgate0); free(wup0); free(wdown0); free(wgate1); free(wup1); free(wdown1);
    free(sgate0); free(sup0); free(sdown0); free(sgate1); free(sup1); free(sdown1);
    unlink(path); rmdir(dir);

    fails += local_fails;
    if(!local_fails) printf("arena realloc guard: ok (slab stays arena-owned, fslab detaches, no interior free, no cross-slot corruption)\n");
    return local_fails;
#undef RCHECK
}

int main(void){
    for(int e=0;e<NE;e++){
        for(int i=0;i<I*D;i++) s_wgate[e][i]=(uint8_t)((e*97+i)&0xFF);
        for(int i=0;i<I*D;i++) s_wup[e][i]=(uint8_t)((e*53+i)&0xFF);
        for(int i=0;i<D*I;i++) s_wdown[e][i]=(uint8_t)((e*31+i)&0xFF);
        /* Distinct, index-carrying patterns per expert/tensor: any cross-slice
         * contamination shows up as a wrong VALUE, not just a wrong count. */
        for(int i=0;i<I;i++) s_sgate[e][i]=bf16_trunc((float)(e*100000+i)*0.001f+0.01f);
        for(int i=0;i<I;i++) s_sup[e][i]  =bf16_trunc((float)(e*200000+i)*0.001f+0.02f);
        for(int i=0;i<D;i++) s_sdown[e][i]=bf16_trunc((float)(e*300000+i)*0.001f+0.03f);
    }

    const char *dir = "tests/tmp_pin_arena_snap";
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir, 0755);
#endif
    char path[300]; snprintf(path,sizeof path,"%s/model.safetensors",dir);

    /* Build the safetensors header for NE experts x (gate/up/down weight + .qs). */
    char hdr[8192]; int hl=0; int64_t off=0;
    hl += snprintf(hdr+hl,sizeof(hdr)-hl,"{");
    int64_t o_wg[NE],o_sg[NE],o_wu[NE],o_su[NE],o_wd[NE],o_sd[NE];
    for(int e=0;e<NE;e++){
        o_wg[e]=off; off+=(int64_t)I*D;
        o_sg[e]=off; off+=(int64_t)I*2;
        o_wu[e]=off; off+=(int64_t)I*D;
        o_su[e]=off; off+=(int64_t)I*2;
        o_wd[e]=off; off+=(int64_t)D*I;
        o_sd[e]=off; off+=(int64_t)D*2;
        hl += snprintf(hdr+hl,sizeof(hdr)-hl,
            "\"model.layers.0.mlp.experts.%d.gate_proj.weight\":{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]},"
            "\"model.layers.0.mlp.experts.%d.gate_proj.weight.qs\":{\"dtype\":\"BF16\",\"shape\":[%d],\"data_offsets\":[%lld,%lld]},"
            "\"model.layers.0.mlp.experts.%d.up_proj.weight\":{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]},"
            "\"model.layers.0.mlp.experts.%d.up_proj.weight.qs\":{\"dtype\":\"BF16\",\"shape\":[%d],\"data_offsets\":[%lld,%lld]},"
            "\"model.layers.0.mlp.experts.%d.down_proj.weight\":{\"dtype\":\"U8\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]},"
            "\"model.layers.0.mlp.experts.%d.down_proj.weight.qs\":{\"dtype\":\"BF16\",\"shape\":[%d],\"data_offsets\":[%lld,%lld]}%s",
            e,I,D,(long long)o_wg[e],(long long)(o_wg[e]+(int64_t)I*D),
            e,I,(long long)o_sg[e],(long long)(o_sg[e]+(int64_t)I*2),
            e,I,D,(long long)o_wu[e],(long long)(o_wu[e]+(int64_t)I*D),
            e,I,(long long)o_su[e],(long long)(o_su[e]+(int64_t)I*2),
            e,D,I,(long long)o_wd[e],(long long)(o_wd[e]+(int64_t)D*I),
            e,D,(long long)o_sd[e],(long long)(o_sd[e]+(int64_t)D*2),
            e==NE-1?"":",");
    }
    hl += snprintf(hdr+hl,sizeof(hdr)-hl,"}");
    if(hl<0 || hl>=(int)sizeof hdr){ printf("FAIL: header too small\n"); return 1; }

    FILE *f=fopen(path,"wb");
    if(!f){ printf("FAIL: cannot create %s (run from c/, like tools/run_tests.py does)\n", path); return 1; }
    uint64_t hlen=(uint64_t)hl;
    fwrite(&hlen,8,1,f); fwrite(hdr,1,hl,f);
    for(int e=0;e<NE;e++){
        fwrite(s_wgate[e],1,(size_t)I*D,f); append_bf16(f,s_sgate[e],I);
        fwrite(s_wup[e],1,(size_t)I*D,f);   append_bf16(f,s_sup[e],I);
        fwrite(s_wdown[e],1,(size_t)D*I,f); append_bf16(f,s_sdown[e],D);
    }
    fclose(f);

    static Model m; memset(&m,0,sizeof m);
    st_init(&m.S, dir);
    m.c.n_layers=1; m.c.hidden=D; m.c.moe_inter=I; m.ebits=8;

    /* Force the arena-bind body to run: it's gated on g_numa_nodes>=2, which
     * real hardware only reports on a genuine multi-socket/multi-node NUMA
     * box. numa_slab_bind's mbind syscall is fire-and-forget (its return value
     * is never checked) -- whether it actually succeeds on this host doesn't
     * affect the C-level buffer-sizing behavior under test here. */
    g_numa_nodes = 2;

    int NR = m.c.n_layers+1;
    m.pin  = calloc((size_t)NR, sizeof(ESlot*));
    m.npin = calloc((size_t)NR, sizeof(int));

    PinRec r[NE]; int slot_of[NE];
    for(int e=0;e<NE;e++){ r[e]=(PinRec){0,e,(uint32_t)(1000-e)}; slot_of[e]=e; }
    m.npin[0]=NE;
    m.pin[0]=calloc(NE, sizeof(ESlot));

    pin_arena_bind(&m, r, slot_of, 0, NE);

    uint8_t *orig_slab[NE]; float *orig_fslab[NE];
    for(int e=0;e<NE;e++){
        CHECK(m.pin[0][e].aslab && m.pin[0][e].afslab);   /* actually arena-bound, not the fallback path */
        orig_slab[e]=m.pin[0][e].slab; orig_fslab[e]=m.pin[0][e].fslab;
    }
    printf("arena bound: %d slots, slab_cap=%" PRId64 " fslab_cap=%" PRId64 " (scale count)\n",
           NE,(int64_t)m.pin[0][0].slab_cap,(int64_t)m.pin[0][0].fslab_cap);
    CHECK(m.pin[0][0].fslab_cap == I+I+D);   /* == qs.NS, not qtot/4 */

    /* Load LAST-to-FIRST -- see the file header comment for why this ordering
     * matters for catching slice-into-neighbour corruption. */
    for(int e=NE-1;e>=0;e--){
        int rc = expert_load_impl(&m, 0, e, &m.pin[0][slot_of[e]], 1, 0);
        CHECK(rc==0);
    }

    /* No realloc should have happened for any slot: the arena was sized
     * correctly upfront (a realloc here would silently mask exactly the kind
     * of sizing bug this task fixes, by falling back to a safe individual
     * allocation instead of crashing). */
    for(int e=0;e<NE;e++){
        CHECK(m.pin[0][e].slab==orig_slab[e]);
        CHECK(m.pin[0][e].fslab==orig_fslab[e]);
    }

    /* Bit-exact scale verification for ALL THREE experts, checked only AFTER
     * every load -- this is what catches "slice runs into neighbour", which a
     * clean ASan run alone cannot: only the LAST slot's overrun reaches
     * unallocated memory, a middle slot's overrun corrupts a neighbour
     * in-place, silently, inside the arena's own valid allocation. */
    for(int e=0;e<NE;e++){
        QT *qt[3] = { &m.pin[0][slot_of[e]].g, &m.pin[0][slot_of[e]].u, &m.pin[0][slot_of[e]].d };
        int ok=1;
        for(int i=0;i<I && ok;i++) if(qt[0]->s[i]!=s_sgate[e][i]){ ok=0;
            printf("FAIL expert %d gate scale[%d]: %.9g want %.9g\n",e,i,(double)qt[0]->s[i],(double)s_sgate[e][i]); fails++; }
        for(int i=0;i<I && ok;i++) if(qt[1]->s[i]!=s_sup[e][i]){ ok=0;
            printf("FAIL expert %d up scale[%d]: %.9g want %.9g\n",e,i,(double)qt[1]->s[i],(double)s_sup[e][i]); fails++; }
        for(int i=0;i<D && ok;i++) if(qt[2]->s[i]!=s_sdown[e][i]){ ok=0;
            printf("FAIL expert %d down scale[%d]: %.9g want %.9g\n",e,i,(double)qt[2]->s[i],(double)s_sdown[e][i]); fails++; }
    }

    unlink(path); rmdir(dir);

    test_arena_realloc_guard();

    if(fails){ printf("pin_arena tests: %d FAILED\n", fails); return 1; }
    printf("pin_arena tests: ok (3-slot arena, BF16 .qs, no realloc, no cross-slice corruption, "
           "arena realloc guard independent per-buffer)\n");
    return 0;
}
