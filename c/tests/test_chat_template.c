/* Chat-template byte-pinning: build_turn_prompt (the engine-side prompt builder used by
 * `coli chat`'s interactive line protocol, run_serve) plus the mt_is_k26 discriminator.
 *
 * Kimi-K2.6 ships a REAL chat_template.jinja that differs from K2-Thinking's in exactly
 * two ways (pinned from the checkpoint's own chat_template.jinja, rendered through
 * transformers' ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True),
 * not hand-traced):
 *   1. it NEVER injects a default "You are Kimi..." system preamble (unlike K2-Thinking's
 *      first-turn-only preamble);
 *   2. it ALWAYS appends a think marker after the assistant turn -- "<think>" when
 *      thinking is on, "<think></think>" when off -- exactly GLM's THINK-env convention,
 *      unlike K2-Thinking, which has no such lever and gets neither marker.
 *
 * Both checkpoints report model_type=="kimi_k2" after the converter's flatten (which
 * deliberately drops K2.6's distinguishing outer model_type "kimi_k25" rather than merging
 * it, to protect the exact-match template selectors -- see convert_fp8_to_int4.py's
 * flatten_container_config docstring). mt_is_k26() discriminates instead on
 * rope_scaling.beta_fast, the one config number verified to differ between the two REAL
 * checkpoints (K2-Thinking: 1.0; K2.6: 32.0) -- see sample.h for the full rationale. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)
#define CHECK_STREQ(a,b) do{ const char *_a=(a),*_b=(b); if(strcmp(_a,_b)!=0){ \
    printf("FAIL %s:%d: got %s\nwant %s\n", __FILE__, __LINE__, _a, _b); fails++; } }while(0)

static jval *parse(const char *src, char **arena){
    char *buf = strdup(src);
    return json_parse(buf, arena);
}

/* Builds a Cfg the way load_cfg would for a K2-family container, distinguished only by
 * rope_scaling.beta_fast -- exactly the real shape difference between K2-Thinking's
 * validated config and K2.6's real source config.json (both verified against real
 * checkpoint files). */
static void cfg_k2(Cfg *c, double beta_fast){
    memset(c, 0, sizeof *c);
    strncpy(c->model_type, "kimi_k2", sizeof c->model_type - 1);
    c->qk_rope = 64; c->theta = 50000.f;
    char *ar = NULL;
    char src[512];
    snprintf(src, sizeof src,
        "{\"rope_scaling\":{\"type\":\"yarn\",\"factor\":64.0,"
        "\"original_max_position_embeddings\":4096,\"beta_fast\":%g,"
        "\"beta_slow\":1.0,\"mscale\":1.0,\"mscale_all_dim\":1.0}}", beta_fast);
    jval *r = parse(src, &ar);
    rope_table_init(c, r);
    free(ar);
}

int main(void){
    /* ---- 1. mt_is_k26 discriminator ---- */
    {
        Cfg k2t; cfg_k2(&k2t, 1.0);
        CHECK(mt_is_k2(&k2t));
        CHECK(!mt_is_k26(&k2t));   /* K2-Thinking: beta_fast==1.0, not K2.6 */

        Cfg k26; cfg_k2(&k26, 32.0);
        CHECK(mt_is_k2(&k26));
        CHECK(mt_is_k26(&k26));   /* K2.6: beta_fast==32.0 */

        Cfg glm; memset(&glm, 0, sizeof glm);
        strncpy(glm.model_type, "glm_moe_dsa", sizeof glm.model_type - 1);
        glm.qk_rope = 64; glm.theta = 10000.f;
        char *ar = NULL; jval *r = parse("{\"hidden_size\":1}", &ar);
        rope_table_init(&glm, r); free(ar);
        CHECK(!mt_is_k2(&glm));
        CHECK(!mt_is_k26(&glm));   /* GLM: no rope_scaling at all -> default beta_fast=1.0 */
    }

    /* ---- 1b. source_variant marker takes precedence over the beta_fast fallback
     * (rope_scaling.beta_fast is a YaRN tuning value with no semantic tie to
     * template choice, so the converter now stamps an explicit "_colibri_source_variant"
     * marker; see sample.h's mt_is_k26 for the full rationale). load_cfg populates
     * Cfg.source_variant from config.json's "_colibri_source_variant" key; these tests
     * set it directly, the same field load_cfg would have set. ---- */
    {
        /* Marker says K2.6 even though beta_fast==1.0 would say K2-Thinking: marker wins. */
        Cfg c; cfg_k2(&c, 1.0);
        strncpy(c.source_variant, "kimi_k25", sizeof c.source_variant - 1);
        CHECK(mt_is_k2(&c));
        CHECK(mt_is_k26(&c));

        /* Marker present but NOT "kimi_k25": decisive as "not K2.6" even though
         * beta_fast==32.0 would say K2.6 via the fallback -- an empty/absent marker is
         * the only thing that falls through to beta_fast, not merely a marker mismatch. */
        Cfg c2; cfg_k2(&c2, 32.0);
        strncpy(c2.source_variant, "something_else", sizeof c2.source_variant - 1);
        CHECK(mt_is_k2(&c2));
        CHECK(!mt_is_k26(&c2));

        /* Empty marker (the K2-Thinking/pre-marker-converter case): falls back to
         * beta_fast, exactly the pre-marker behavior -- regression guard. */
        Cfg c3; cfg_k2(&c3, 32.0);
        CHECK(c3.source_variant[0] == 0);
        CHECK(mt_is_k26(&c3));
    }

    /* ---- 2. K2-Thinking regression: unchanged byte-for-byte (is_k26=0) ---- */
    {
        Cfg c; cfg_k2(&c, 1.0);
        int is_k2 = mt_is_k2(&c), is_k26 = mt_is_k26(&c);
        CHECK(!is_k26);
        char buf[1<<12];
        int bl = build_turn_prompt(buf, sizeof buf, is_k2, is_k26, 1, 1, "hello", "<think></think>");
        buf[bl] = 0;
        CHECK_STREQ(buf,
            "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by "
            "Moonshot AI.<|im_end|><|im_user|>user<|im_middle|>hello<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|>");
        /* THINK must have NO effect on K2-Thinking (tk ignored on this path) */
        int bl2 = build_turn_prompt(buf, sizeof buf, is_k2, is_k26, 1, 1, "hello", "<think>");
        buf[bl2] = 0;
        CHECK_STREQ(buf,
            "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by "
            "Moonshot AI.<|im_end|><|im_user|>user<|im_middle|>hello<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|>");
        /* first=0 (not turn 1): no preamble */
        int bl3 = build_turn_prompt(buf, sizeof buf, is_k2, is_k26, 1, 0, "hello", "<think></think>");
        buf[bl3] = 0;
        CHECK_STREQ(buf,
            "<|im_user|>user<|im_middle|>hello<|im_end|><|im_assistant|>assistant<|im_middle|>");
    }

    /* ---- 3. K2.6, thinking OFF: pinned against transformers' Jinja2 render of the real
     * K2.6 chat_template.jinja for messages=[{"role":"user","content":"hello"}],
     * add_generation_prompt=True, thinking=False. ---- */
    {
        Cfg c; cfg_k2(&c, 32.0);
        int is_k2 = mt_is_k2(&c), is_k26 = mt_is_k26(&c);
        CHECK(is_k2); CHECK(is_k26);
        char buf[1<<12];
        int bl = build_turn_prompt(buf, sizeof buf, is_k2, is_k26, 1, 1, "hello", "<think></think>");
        buf[bl] = 0;
        CHECK_STREQ(buf,
            "<|im_user|>user<|im_middle|>hello<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|><think></think>");
        /* first=0 must render identically -- K2.6's template has no first-turn-only
         * preamble to begin with, unlike K2-Thinking's */
        int bl2 = build_turn_prompt(buf, sizeof buf, is_k2, is_k26, 1, 0, "hello", "<think></think>");
        buf[bl2] = 0;
        CHECK_STREQ(buf,
            "<|im_user|>user<|im_middle|>hello<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|><think></think>");
    }

    /* ---- 4. K2.6, thinking ON: same real-template pin, thinking=True ---- */
    {
        Cfg c; cfg_k2(&c, 32.0);
        int is_k2 = mt_is_k2(&c), is_k26 = mt_is_k26(&c);
        char buf[1<<12];
        int bl = build_turn_prompt(buf, sizeof buf, is_k2, is_k26, 1, 1, "hello", "<think>");
        buf[bl] = 0;
        CHECK_STREQ(buf,
            "<|im_user|>user<|im_middle|>hello<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|><think>");
    }

    /* ---- 5. GLM regression: is_k2=0 path is untouched by the is_k26 parameter ---- */
    {
        char buf[1<<12];
        int bl = build_turn_prompt(buf, sizeof buf, 0, 0, 1, 1, "hello", "<think></think>");
        buf[bl] = 0;
        CHECK_STREQ(buf, "[gMASK]<sop><|user|>hello<|assistant|><think></think>");
        int bl2 = build_turn_prompt(buf, sizeof buf, 0, 0, 1, 1, "hello", "<think>");
        buf[bl2] = 0;
        CHECK_STREQ(buf, "[gMASK]<sop><|user|>hello<|assistant|><think>");
        int bl3 = build_turn_prompt(buf, sizeof buf, 0, 0, 1, 0, "hello", "<think></think>");
        buf[bl3] = 0;
        CHECK_STREQ(buf, "<|user|>hello<|assistant|><think></think>");
    }

    if(fails){ printf("chat template tests: %d FAILED\n", fails); return 1; }
    printf("chat template tests: ok (mt_is_k26 discriminator, K2-Thinking regression, "
           "K2.6 think/nothink pinned against the real chat_template.jinja, GLM regression)\n");
    return 0;
}
