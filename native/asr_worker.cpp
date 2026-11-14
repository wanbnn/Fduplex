// Persistent PCM worker. Protocol: command byte + LE uint32 sample count + f32 PCM.
// Responses are one JSON record per request. Native diagnostics go to stderr.
#include "transcribe.h"
#include "transcribe/parakeet.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <unistd.h>

static std::string quote(const char *value) {
    std::string out = "\"";
    for (const unsigned char c : std::string(value ? value : "")) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c < 32) { char escape[7]; std::snprintf(escape, sizeof escape, "\\u%04x", c); out += escape; }
        else out += c;
    }
    return out + '"';
}
int main(int argc, char **argv) {
    if (argc != 5) { std::fprintf(stderr, "usage: worker model backend language right-context\n"); return 2; }
    FILE *out = fdopen(dup(STDOUT_FILENO), "w");
    dup2(STDERR_FILENO, STDOUT_FILENO);
    auto send = [&](const std::string &json) { std::fprintf(out, "%s\n", json.c_str()); std::fflush(out); };
    auto check = [&](transcribe_status status) {
        if (status == TRANSCRIBE_OK) return true;
        send("{\"error\":" + quote(transcribe_status_string(status)) + "}"); return false;
    };
    transcribe_model_load_params load;
    transcribe_model_load_params_init(&load);
    const std::string backend(argv[2]);
    if (backend == "rocm") load.backend = TRANSCRIBE_BACKEND_ROCM;
    else if (backend == "cpu") load.backend = TRANSCRIBE_BACKEND_CPU;
    else if (backend == "vulkan") load.backend = TRANSCRIBE_BACKEND_VULKAN;
    else if (backend == "cuda") load.backend = TRANSCRIBE_BACKEND_CUDA;
    else if (backend == "metal") load.backend = TRANSCRIBE_BACKEND_METAL;
    else if (backend != "auto") { send("{\"error\":\"backend inválido\"}"); return 2; }
    transcribe_session *session = nullptr;
    if (!check(transcribe_open(argv[1], &load, nullptr, &session))) return 1;
    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    if (!check(transcribe_model_get_capabilities(transcribe_get_model(session), &caps))) {
        transcribe_session_free(session); return 1;
    }
    send(std::string("{\"ready\":true,\"streaming\":") + (caps.supports_streaming ? "true" : "false") +
         ",\"backend\":" + quote(transcribe_model_backend(transcribe_get_model(session))) + "}");
    transcribe_run_params run;
    transcribe_run_params_init(&run);
    std::string language(argv[3]);
    bool exact_language = false;
    for (int i = 0; i < caps.n_languages; ++i) exact_language |= language == caps.languages[i];
    if (!exact_language && language.find('-') != std::string::npos) {
        const auto short_language = language.substr(0, language.find('-'));
        for (int i = 0; i < caps.n_languages; ++i)
            if (short_language == caps.languages[i]) { language = short_language; break; }
    }
    run.language = language == "auto" ? "auto" : language.c_str();
    transcribe_stream_params stream;
    transcribe_stream_params_init(&stream);
    transcribe_parakeet_stream_ext ext;
    transcribe_parakeet_stream_ext_init(&ext);
    if (transcribe_model_accepts_ext_kind(transcribe_get_model(session), TRANSCRIBE_EXT_SLOT_STREAM, TRANSCRIBE_EXT_KIND_PARAKEET_STREAM)) {
        ext.att_context_right = std::atoi(argv[4]);
        stream.family = &ext.ext;
    }
    auto text = [&]() {
        transcribe_stream_text view;
        transcribe_stream_text_init(&view);
        if (check(transcribe_stream_get_text(session, &view)))
            send("{\"text\":" + quote(view.full_text) + ",\"committed\":" + quote(view.committed_text) +
                 ",\"tentative\":" + quote(view.tentative_text) + "}");
    };
    while (true) {
        unsigned char cmd;
        unsigned char header[4];
        if (std::fread(&cmd, 1, 1, stdin) != 1 || std::fread(header, 1, 4, stdin) != 4) break;
        const uint32_t count = uint32_t(header[0]) | uint32_t(header[1]) << 8 | uint32_t(header[2]) << 16 | uint32_t(header[3]) << 24;
        if (count > 16000 * 60) { send("{\"error\":\"PCM excede 60 segundos\"}"); break; }
        std::vector<float> pcm(count);
        if (count && std::fread(pcm.data(), sizeof(float), count, stdin) != count) break;
        if (cmd == 'Q') break;
        if (cmd == 'B') {
            transcribe_stream_reset(session);
            if (check(transcribe_stream_begin(session, &run, &stream))) send("{\"ok\":true}");
        } else if (cmd == 'A') {
            if (check(transcribe_stream_feed(session, pcm.data(), count, nullptr))) text();
        } else if (cmd == 'F') {
            if (check(transcribe_stream_finalize(session, nullptr))) text();
        } else if (cmd == 'R') {
            if (check(transcribe_run(session, pcm.data(), count, &run)))
                send("{\"text\":" + quote(transcribe_full_text(session)) + "}");
        } else send("{\"error\":\"comando inválido\"}");
    }
    transcribe_session_free(session);
    std::fclose(out);
}
