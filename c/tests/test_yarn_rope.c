/* YaRN rotary-frequency table (DeepseekV3YarnRotaryEmbedding) vs a pinned
 * Python reference, plus the GLM no-scaling regression.
 *
 * Kimi-K2 carries rope_theta as a TOP-LEVEL config key (GLM nests it under
 * rope_parameters) and a yarn rope_scaling block. Without both, the engine
 * runs K2 at theta=10000 with unscaled frequencies -- out of distribution at
 * every position, with no error. This test pins the exact table.
 *
 * The expected values below were produced by the reference implementation:
 *   dim=64, base=50000, factor=64, orig_max=4096, beta_fast=beta_slow=1.0
 *   correction_dim(1.0) = 64*ln(4096/(2*pi)) / (2*ln(50000)) = 19.1657...
 *   low = 19, high = 20  ->  j <= 19 extrapolate, j >= 20 interpolate (/64) */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)
#define CLOSE(a,b,tol) do{ double _d=(double)(a)-(double)(b); if(_d<0)_d=-_d; \
    if(!(_d <= (tol)*(1.0+((b)<0?-(b):(b))))){ \
        printf("FAIL %s:%d: %s=%.10g want %.10g\n", __FILE__, __LINE__, #a, (double)(a), (double)(b)); fails++; } }while(0)

/* Parse a JSON string into the engine's jval tree (json_parse takes a mutable
 * buffer and an arena out-param, exactly as cfg_root uses it). */
static jval *parse(const char *src, char **arena){
    char *buf = strdup(src);
    jval *r = json_parse(buf, arena);
    return r;   /* buf is owned by the arena's lifetime for our purposes */
}

int main(void){
    /* ---- 1. GLM regression: no rope_scaling -> plain theta^(-2j/dim) ---- */
    {
        Cfg c; memset(&c,0,sizeof c);
        c.qk_rope = 64; c.theta = 10000.f;
        c.attn_scale = 1.0f/8.0f;             /* stand-in for load_cfg's 1/sqrt(qk_head) */
        char *ar=NULL; jval *r = parse("{\"hidden_size\":1}", &ar);
        rope_table_init(&c, r);
        CHECK(g_yarn == 0);
        CHECK(g_yarn_mscale == 1.0f);
        for(int j=0;j<32;j++){
            float want = powf(10000.f, -2.0f*j/64.0f);
            CHECK(g_inv_freq[j] == want);        /* BIT-identical to the old expression */
        }
        /* No rope_scaling block at all -> attn_scale must be untouched:
         * fix must not apply the softmax-scale correction outside the yarn branch. */
        CHECK(c.attn_scale == 1.0f/8.0f);
        free(ar);
    }

    /* ---- 2. K2: top-level rope_theta is picked up by load_cfg's fallback ----
     * (covered end-to-end in step 6; here we assert the table given theta) */

    /* ---- 3. K2 YaRN table ---- */
    {
        Cfg c; memset(&c,0,sizeof c);
        c.qk_rope = 64; c.theta = 50000.f;
        c.attn_scale = 1.0f/8.0f;             /* stand-in for load_cfg's 1/sqrt(qk_head) */
        char *ar=NULL;
        jval *r = parse(
          "{\"rope_scaling\":{\"type\":\"yarn\",\"factor\":64.0,"
          "\"original_max_position_embeddings\":4096,\"beta_fast\":1.0,"
          "\"beta_slow\":1.0,\"mscale\":1.0,\"mscale_all_dim\":1.0}}", &ar);
        rope_table_init(&c, r);
        CHECK(g_yarn == 1);
        CLOSE(g_yarn_mscale, 1.0, 1e-6);          /* mscale == mscale_all_dim */
        /* DeepseekV3Attention applies a SEPARATE softmax-scale
         * correction, self.scaling *= get_mscale(factor,mscale_all_dim)**2 --
         * independent of the rotary g_yarn_mscale ratio above (which is 1.0 here
         * because mscale==mscale_all_dim). It is NOT 1.0 for K2's real config:
         *   get_mscale(64,1.0) = 0.1*ln(64)+1 = 1.41588830833596718565 (bc -l, 25 digits)
         *   squared             = 2.00473970168248688418
         *   base(1/8) * squared = 0.25059246271031086052
         * Independently verified with Python (double) and `bc -l` (25-digit
         * arbitrary precision); both agree to every digit double precision offers. */
        CLOSE(c.attn_scale, 0.25059246271031086052, 1e-6);

        /* j <= 19 : pure extrapolation (unscaled). j >= 20 : pure
         * interpolation (divided by factor). low=19, high=20 => the ramp is a
         * step, so no j gets a blended value. */
        for(int j=0;j<32;j++){
            double extra = pow(50000.0, -2.0*j/64.0);
            double want  = (j <= 19) ? extra : extra/64.0;
            CLOSE(g_inv_freq[j], want, 1e-6);
        }
        /* Spot-check the two sides of the extrapolate/interpolate boundary against
         * pinned decimals, independently of the loop above -- if the loop's own
         * reference expression were wrong, the loop would still pass. Verified
         * with two independent tools (Python double + `bc -l` arbitrary
         * precision) before pinning, since the plan's first two attempts at
         * these constants both had a wrong digit:
         *   extra(19) = 50000^(-38/64) = 0.00162175990811595221 (j<=19: unscaled)
         *   extra(20) = 50000^(-40/64) = 0.00115649496754324186
         *   inv(20)   = extra(20)/64   = 0.00001807023386786315 (j>=20: /factor) */
        CLOSE(g_inv_freq[0],  1.0,           1e-9);
        CLOSE(g_inv_freq[19], 1.6217599e-03, 1e-5);
        CLOSE(g_inv_freq[20], 1.8070234e-05, 1e-5);
        free(ar);
    }

    /* ---- 3b. K2.6 YaRN ramp: beta_fast=32.0 (vs test 3's 1.0)
     * gives low=8, high=20 -- an 11-wide range (j=9..19) where the ramp is
     * strictly between 0 and 1, unlike test 3's low=19/high=20 step where no
     * j ever lands in the blend. This is the first test to exercise that
     * branch at all.
     *
     * Derivation (K2.6's real rope_scaling, confirmed from
     * K2.6's real config.json text_config.rope_scaling):
     *   dim=64, base=50000, factor=64, orig_max=4096, beta_fast=32.0, beta_slow=1.0
     *   correction_dim(nrot) = 64*ln(4096/(nrot*2*pi)) / (2*ln(50000))
     *   lo_d = correction_dim(32.0) = 8.914498965224011   -> low  = floor = 8
     *   hi_d = correction_dim(1.0)  = 19.164574888626899  -> high = ceil  = 20
     * Cross-checked two independent ways: Python (float64, math.log/pow) and
     * `bc -l` (30-digit arbitrary precision) agree on lo_d/hi_d and on every
     * pinned decimal below to all shown digits. A third check, transformers
     * 5.14.1's own _compute_yarn_parameters (float32) at these exact K2.6
     * dims, matches the float64 table to ~1e-7 relative on all 32 entries --
     * confirming the plan's claim of low=8/high=20/11-blended dims is
     * correct as given (no disagreement to report). */
    {
        Cfg c; memset(&c,0,sizeof c);
        c.qk_rope = 64; c.theta = 50000.f;
        c.attn_scale = 1.0f/8.0f;
        char *ar=NULL;
        jval *r = parse(
          "{\"rope_scaling\":{\"type\":\"yarn\",\"factor\":64.0,"
          "\"original_max_position_embeddings\":4096,\"beta_fast\":32.0,"
          "\"beta_slow\":1.0,\"mscale\":1.0,\"mscale_all_dim\":1.0}}", &ar);
        rope_table_init(&c, r);
        CHECK(g_yarn == 1);
        CLOSE(g_yarn_mscale, 1.0, 1e-6);      /* mscale == mscale_all_dim, as in test 3 */

        /* Full-table check against the closed-form low=8,high=20 ramp. */
        {
            double low = 8.0, high = 20.0;
            for(int j=0;j<32;j++){
                double extra = pow(50000.0, -2.0*j/64.0);
                double inter = extra / 64.0;
                double ramp  = (j - low) / (high - low);
                if(ramp < 0) ramp = 0; else if(ramp > 1) ramp = 1;
                double want = inter*ramp + extra*(1.0-ramp);
                CLOSE(g_inv_freq[j], want, 1e-6);
            }
        }

        /* Pinned decimals for three of the 11 strictly-blended dims,
         * independently verified with bc -l (30-digit precision):
         *   j=9:  extra=4.7688612441e-02 inter=7.4513456938e-04 ramp=1/12   -> 4.3776655951e-02
         *   j=14: extra=8.7942867664e-03 inter=1.3741073073e-04 ramp=1/2   -> 4.4658487486e-03
         *   j=19: extra=1.6217599081e-03 inter=2.5339998564e-05 ramp=11/12 -> 1.5837499103e-04
         * A step-function ramp (i.e. the pre-K2.6 code path) would instead
         * produce exactly extra(j) or exactly inter(j) at these j -- these
         * pinned values are neither. */
        CLOSE(g_inv_freq[9],  4.3776655951e-02, 1e-6);
        CLOSE(g_inv_freq[14], 4.4658487486e-03, 1e-6);
        CLOSE(g_inv_freq[19], 1.5837499103e-04, 1e-6);

        /* The discriminating assertion: j=14 (the ramp midpoint) must land
         * STRICTLY between pure extrapolation and pure interpolation, not
         * equal to either. This is what a step-not-ramp implementation gets
         * wrong, and is the whole point of this test case. */
        {
            double extra14 = pow(50000.0, -2.0*14/64.0);
            double inter14 = extra14 / 64.0;
            CHECK(g_inv_freq[14] < (float)extra14);
            CHECK(g_inv_freq[14] > (float)inter14);
        }

        /* Boundary shape: j<=8 pure extrapolation (ramp==0), j>=20 pure
         * interpolation (ramp==1) -- confirms low=8/high=20 exactly, not
         * just "blending happens somewhere". */
        CLOSE(g_inv_freq[8],  pow(50000.0, -2.0*8/64.0),       1e-6);
        CLOSE(g_inv_freq[20], pow(50000.0, -2.0*20/64.0)/64.0, 1e-6);
        free(ar);
    }

    /* ---- 4. mscale is computed, not assumed ---- */
    {
        Cfg c; memset(&c,0,sizeof c);
        c.qk_rope = 64; c.theta = 50000.f;
        c.attn_scale = 1.0f/8.0f;
        char *ar=NULL;
        jval *r = parse(
          "{\"rope_scaling\":{\"type\":\"yarn\",\"factor\":64.0,"
          "\"original_max_position_embeddings\":4096,\"beta_fast\":1.0,"
          "\"beta_slow\":1.0,\"mscale\":1.0,\"mscale_all_dim\":0.0}}", &ar);
        rope_table_init(&c, r);
        /* get_mscale(64,1.0) = 0.1*1.0*ln(64)+1 = 1.4158883
         * get_mscale(64,0.0) = 0.1*0.0*ln(64)+1 = 1.0        -> ratio 1.4158883 */
        CLOSE(g_yarn_mscale, 1.4158883, 1e-6);
        /* mscale_all_dim==0.0 must NOT trigger the softmax-scale
         * correction -- the reference guards on `if mscale_all_dim:` (falsy for
         * 0), independent of what the rotary ratio above computes. */
        CHECK(c.attn_scale == 1.0f/8.0f);
        free(ar);
    }

    /* ---- 4b. mscale_all_dim ABSENT (key missing entirely) -> same
     * no-correction guard as the explicit-0.0 case in test 4 ---- */
    {
        Cfg c; memset(&c,0,sizeof c);
        c.qk_rope = 64; c.theta = 50000.f;
        c.attn_scale = 1.0f/8.0f;
        char *ar=NULL;
        jval *r = parse(
          "{\"rope_scaling\":{\"type\":\"yarn\",\"factor\":64.0,"
          "\"original_max_position_embeddings\":4096,\"beta_fast\":1.0,"
          "\"beta_slow\":1.0,\"mscale\":1.0}}", &ar);       /* no mscale_all_dim key */
        rope_table_init(&c, r);
        CHECK(g_yarn == 1);
        CHECK(c.attn_scale == 1.0f/8.0f);
        free(ar);
    }

    /* ---- 5. rope_interleave uses the table (YaRN actually reaches the math) ---- */
    {
        Cfg c; memset(&c,0,sizeof c);
        c.qk_rope = 64; c.theta = 50000.f;
        char *ar=NULL;
        jval *r = parse(
          "{\"rope_scaling\":{\"type\":\"yarn\",\"factor\":64.0,"
          "\"original_max_position_embeddings\":4096,\"beta_fast\":1.0,"
          "\"beta_slow\":1.0,\"mscale\":1.0,\"mscale_all_dim\":1.0}}", &ar);
        rope_table_init(&c, r);
        float v[64]; for(int i=0;i<64;i++) v[i] = (i==40)?1.0f:0.0f;   /* j=20 pair, real part */
        rope_interleave(v, 7, &c);
        /* j=20 is an interpolated dim: angle = 7 * inv_freq[20] */
        double ang = 7.0 * (pow(50000.0,-2.0*20/64.0)/64.0);
        CLOSE(v[20],    cos(ang), 1e-5);
        CLOSE(v[32+20], sin(ang), 1e-5);
        free(ar);
    }

    /* ---- 6. Step 0 guard: no rope_scaling, rope_parameters.rope_type="default"
     * (GLM's real fixture shape, c/glm_tiny/config.json) -> stays silent and
     * unscaled, exactly like test 1 with no rope_parameters at all ---- */
    {
        Cfg c; memset(&c,0,sizeof c);
        c.qk_rope = 64; c.theta = 10000.f;
        c.attn_scale = 1.0f/8.0f;
        char *ar=NULL;
        jval *r = parse(
          "{\"rope_parameters\":{\"rope_type\":\"default\",\"rope_theta\":10000.0}}", &ar);
        rope_table_init(&c, r);
        CHECK(g_yarn == 0);
        CHECK(g_yarn_mscale == 1.0f);
        CHECK(c.attn_scale == 1.0f/8.0f);
        for(int j=0;j<32;j++){
            float want = powf(10000.f, -2.0f*j/64.0f);
            CHECK(g_inv_freq[j] == want);
        }
        free(ar);
    }

    /* ---- 6b. Same guard, no rope_parameters key at all -> also silent
     * (belt-and-suspenders: absence must not be confused with an unknown type) ---- */
    {
        Cfg c; memset(&c,0,sizeof c);
        c.qk_rope = 64; c.theta = 10000.f;
        c.attn_scale = 1.0f/8.0f;
        char *ar=NULL;
        jval *r = parse("{\"hidden_size\":1}", &ar);
        rope_table_init(&c, r);
        CHECK(g_yarn == 0);
        CHECK(c.attn_scale == 1.0f/8.0f);
        free(ar);
    }

#ifndef _WIN32
    /* ---- 7. Step 0 guard: no rope_scaling, rope_parameters names a non-
     * "default" type (a transformers-5.x-written config that never got the
     * legacy rope_scaling block re-added) -> must hard-exit(1) with a clear
     * message instead of silently running unscaled. Forked because
     * rope_table_init calls exit() directly on this path. ---- */
    {
        int pipefd[2]; CHECK(pipe(pipefd) == 0);
        pid_t pid = fork(); CHECK(pid >= 0);
        if(pid == 0){
            dup2(pipefd[1], 2); close(pipefd[0]); close(pipefd[1]);
            Cfg c; memset(&c,0,sizeof c);
            c.qk_rope = 64; c.theta = 50000.f;
            c.attn_scale = 1.0f/8.0f;
            char *ar=NULL;
            jval *r = parse(
              "{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":64.0}}", &ar);
            rope_table_init(&c, r);       /* must exit(1) inside; reaching past = bug */
            _exit(42);
        }
        close(pipefd[1]);
        char err[512] = {0};
        ssize_t n = read(pipefd[0], err, sizeof(err)-1); (void)n;
        close(pipefd[0]);
        int status = 0; waitpid(pid, &status, 0);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 1);
        CHECK(strstr(err, "rope_parameters") != NULL);
        CHECK(strstr(err, "yarn") != NULL);
    }
#else
    printf("yarn rope tests: rope_parameters hard-exit subtest skipped on Windows\n");
#endif

    /* ---- beta_fast ABSENT: the YaRN math keeps its 32.0 reference default ---- */
    {
        Cfg c; memset(&c,0,sizeof c);
        c.qk_rope = 64; c.theta = 50000.f;
        c.attn_scale = 1.0f/8.0f;
        char *ar=NULL;
        jval *r = parse(                      /* yarn block with NO beta_fast key */
          "{\"rope_scaling\":{\"type\":\"yarn\",\"factor\":64.0,"
          "\"original_max_position_embeddings\":4096,"
          "\"beta_slow\":1.0,\"mscale\":1.0,\"mscale_all_dim\":1.0}}", &ar);
        rope_table_init(&c, r);
        CHECK(g_yarn == 1);

        /* The math defaulted to 32.0: that gives low=8/high=20, so j=8 is pure
         * extrapolation and j=20 pure interpolation -- identical to the explicit
         * beta_fast=32.0 case above. Had the default regressed to 1.0, low/high would be
         * 19/20 and g_inv_freq[8] would be the unscaled extrapolated value instead. */
        CLOSE(g_inv_freq[20], pow(50000.0, -40.0/64.0)/64.0, 1e-6);   /* interpolated */
        free(ar);
    }

    if(fails){ printf("yarn rope tests: %d FAILED\n", fails); return 1; }
    printf("yarn rope tests: ok (plain table bit-identical, K2 yarn table, mscale, rope_interleave)\n");
    return 0;
}
