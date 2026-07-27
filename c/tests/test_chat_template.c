/* Chat-template byte-pinning: build_turn_prompt (the engine-side prompt builder used by
 * `coli chat`'s interactive line protocol, run_serve) plus the mt_is_k2 arch check.
 *
 * Kimi-K2.6's template is pinned from the checkpoint's own chat_template.jinja,
 * rendered through transformers' ImmutableSandboxedEnvironment(trim_blocks=True,
 * lstrip_blocks=True): no session prefix, no default system preamble, and a think
 * marker always appended after the assistant turn -- "<think>" when thinking is on,
 * "<think></think>" when off (GLM's THINK-env convention). */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)
#define CHECK_STREQ(a,b) do{ const char *_a=(a),*_b=(b); if(strcmp(_a,_b)!=0){ \
    printf("FAIL %s:%d: got %s\nwant %s\n", __FILE__, __LINE__, _a, _b); fails++; } }while(0)

static void cfg_k2(Cfg *c){
    memset(c, 0, sizeof *c);
    strncpy(c->model_type, "kimi_k2", sizeof c->model_type - 1);
}

int main(void){
    /* ---- 1. mt_is_k2 arch check ---- */
    {
        Cfg k2; cfg_k2(&k2);
        CHECK(mt_is_k2(&k2));

        Cfg glm; memset(&glm, 0, sizeof glm);
        strncpy(glm.model_type, "glm_moe_dsa", sizeof glm.model_type - 1);
        CHECK(!mt_is_k2(&glm));
    }

    /* ---- 2. K2, thinking OFF: pinned against transformers' Jinja2 render of the real
     * K2.6 chat_template.jinja for messages=[{"role":"user","content":"hello"}],
     * add_generation_prompt=True, thinking=False. ---- */
    {
        Cfg c; cfg_k2(&c);
        int is_k2 = mt_is_k2(&c);
        CHECK(is_k2);
        char buf[1<<12];
        int bl = build_turn_prompt(buf, sizeof buf, is_k2, 1, 1, "hello", "<think></think>");
        buf[bl] = 0;
        CHECK_STREQ(buf,
            "<|im_user|>user<|im_middle|>hello<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|><think></think>");
        /* first=0 must render identically -- the template has no first-turn-only prefix */
        int bl2 = build_turn_prompt(buf, sizeof buf, is_k2, 1, 0, "hello", "<think></think>");
        buf[bl2] = 0;
        CHECK_STREQ(buf,
            "<|im_user|>user<|im_middle|>hello<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|><think></think>");
    }

    /* ---- 3. K2, thinking ON: same real-template pin, thinking=True ---- */
    {
        Cfg c; cfg_k2(&c);
        int is_k2 = mt_is_k2(&c);
        char buf[1<<12];
        int bl = build_turn_prompt(buf, sizeof buf, is_k2, 1, 1, "hello", "<think>");
        buf[bl] = 0;
        CHECK_STREQ(buf,
            "<|im_user|>user<|im_middle|>hello<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|><think>");
    }

    /* ---- 4. GLM regression: is_k2=0 path unchanged ---- */
    {
        char buf[1<<12];
        int bl = build_turn_prompt(buf, sizeof buf, 0, 1, 1, "hello", "<think></think>");
        buf[bl] = 0;
        CHECK_STREQ(buf, "[gMASK]<sop><|user|>hello<|assistant|><think></think>");
        int bl2 = build_turn_prompt(buf, sizeof buf, 0, 1, 1, "hello", "<think>");
        buf[bl2] = 0;
        CHECK_STREQ(buf, "[gMASK]<sop><|user|>hello<|assistant|><think>");
        int bl3 = build_turn_prompt(buf, sizeof buf, 0, 1, 0, "hello", "<think></think>");
        buf[bl3] = 0;
        CHECK_STREQ(buf, "<|user|>hello<|assistant|><think></think>");
    }

    if(fails){ printf("chat template tests: %d FAILED\n", fails); return 1; }
    printf("chat template tests: ok (mt_is_k2, K2 think/nothink pinned against the real "
           "chat_template.jinja, GLM regression)\n");
    return 0;
}
