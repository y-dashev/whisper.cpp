#include "common-sdl.h"
#include "whisper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <poll.h>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

struct ptt_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t capture_id    = -1;
    int32_t max_record_ms = 120000;

    bool use_gpu    = true;
    bool flash_attn = true;

    std::string language = "bg";
    std::string model    = "models/ggml-large-v3-turbo-q5_0.bin";
    std::string output;
};

class terminal_raw_mode {
public:
    bool enable() {
        if (tcgetattr(STDIN_FILENO, &original_) != 0) {
            return false;
        }

        struct termios raw = original_;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN]  = 0;
        raw.c_cc[VTIME] = 0;

        enabled_ = tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) == 0;
        return enabled_;
    }

    ~terminal_raw_mode() {
        if (enabled_) {
            tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_);
        }
    }

private:
    struct termios original_ {};
    bool enabled_ = false;
};

static void print_usage(const char * program, const ptt_params & params) {
    fprintf(stderr, "usage: %s [options]\n\n", program);
    fprintf(stderr, "  -h,      --help              show this help message\n");
    fprintf(stderr, "  -m FILE, --model FILE        model path [%s]\n", params.model.c_str());
    fprintf(stderr, "  -l LANG, --language LANG     bg or en [%s]\n", params.language.c_str());
    fprintf(stderr, "  -f FILE, --file FILE         append transcript to FILE\n");
    fprintf(stderr, "  -t N,    --threads N         inference threads [%d]\n", params.n_threads);
    fprintf(stderr, "  -c ID,   --capture ID        microphone device ID [%d]\n", params.capture_id);
    fprintf(stderr, "            --max-record N     maximum recording length in seconds [%d]\n", params.max_record_ms / 1000);
    fprintf(stderr, "  -ng,     --no-gpu            disable GPU inference\n");
    fprintf(stderr, "  -nfa,    --no-flash-attn     disable flash attention\n");
}

static bool parse_args(int argc, char ** argv, ptt_params & params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: missing value for %s\n", arg.c_str());
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0], params);
            return false;
        } else if (arg == "-m" || arg == "--model") {
            const char * value = next(); if (!value) return false; params.model = value;
        } else if (arg == "-l" || arg == "--language") {
            const char * value = next(); if (!value) return false; params.language = value;
        } else if (arg == "-f" || arg == "--file") {
            const char * value = next(); if (!value) return false; params.output = value;
        } else if (arg == "-t" || arg == "--threads") {
            const char * value = next(); if (!value) return false; params.n_threads = std::stoi(value);
        } else if (arg == "-c" || arg == "--capture") {
            const char * value = next(); if (!value) return false; params.capture_id = std::stoi(value);
        } else if (arg == "--max-record") {
            const char * value = next(); if (!value) return false; params.max_record_ms = 1000 * std::stoi(value);
        } else if (arg == "-ng" || arg == "--no-gpu") {
            params.use_gpu = false;
        } else if (arg == "-nfa" || arg == "--no-flash-attn") {
            params.flash_attn = false;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0], params);
            return false;
        }
    }
    return true;
}

static char read_key(int timeout_ms) {
    struct pollfd input { STDIN_FILENO, POLLIN, 0 };
    if (poll(&input, 1, timeout_ms) <= 0 || !(input.revents & POLLIN)) {
        return 0;
    }

    char key = 0;
    return read(STDIN_FILENO, &key, 1) == 1 ? key : 0;
}

static void progress_callback(struct whisper_context *, struct whisper_state *, int progress, void * user_data) {
    static_cast<std::atomic<int> *>(user_data)->store(progress);
}

static bool styled_terminal() {
    return isatty(STDERR_FILENO) && getenv("NO_COLOR") == nullptr;
}

static const char * language_name(const std::string & language) {
    if (language == "bg") return "Bulgarian";
    if (language == "en") return "English";
    return "Bulgarian";
}

static void clear_screen() {
    if (styled_terminal()) {
        fprintf(stderr, "\033[2J\033[H");
    } else {
        fprintf(stderr, "\n");
    }
}

static const char * clear_line() {
    return styled_terminal() ? "\r\033[2K" : "\r";
}

static void print_dashboard(
        const std::string & language,
        const std::string & transcription,
        const char * status,
        const char * status_color,
        const std::string & notice = {}) {
    const bool styled = styled_terminal();
    const char * cyan   = styled ? "\033[1;36m" : "";
    const char * dim    = styled ? "\033[2m" : "";
    const char * bold   = styled ? "\033[1m" : "";
    const char * reset  = styled ? "\033[0m" : "";
    const char * color  = styled ? status_color : "";

    clear_screen();
    fprintf(stderr, "%s╭────────────────────────────────────────────────────────────╮%s\n", cyan, reset);
    fprintf(stderr, "%s│  🎙  WHISPER DICTATE                                      │%s\n", cyan, reset);
    fprintf(stderr, "%s╰────────────────────────────────────────────────────────────╯%s\n", cyan, reset);
    fprintf(stderr, "\n  %sLanguage%s  %s\n", dim, reset, language_name(language));
    fprintf(stderr, "  %sModel%s     Large V3 Turbo Q5\n", dim, reset);
    fprintf(stderr, "  %sStatus%s    %s%s%s\n", dim, reset, color, status, reset);

    if (!transcription.empty()) {
        fprintf(stderr, "\n%s╭─ Last transcription ──────────────────────────────────────╮%s\n", cyan, reset);
        fprintf(stderr, "  %s\n", transcription.c_str());
        fprintf(stderr, "%s╰────────────────────────────────────────────────────────────╯%s\n", cyan, reset);
    }

    if (!notice.empty()) {
        fprintf(stderr, "\n  %s✓ %s%s\n", bold, notice.c_str(), reset);
    }

    fprintf(stderr, "\n  %sSPACE%s Record     %sB%s Bulgarian     %sE%s English\n",
            bold, reset, bold, reset, bold, reset);
    fprintf(stderr, "  %sC%s     Copy last  %sQ%s Quit\n\n", bold, reset, bold, reset);
    fflush(stderr);
}

static bool copy_to_clipboard(const std::string & text) {
    FILE * clipboard = popen("/usr/bin/pbcopy", "w");
    if (!clipboard) {
        return false;
    }

    const bool written = fwrite(text.data(), 1, text.size(), clipboard) == text.size();
    return pclose(clipboard) == 0 && written;
}

int main(int argc, char ** argv) {
    ptt_params params;
    if (!parse_args(argc, argv, params)) {
        return argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") ? 0 : 1;
    }

    if (params.language != "bg" && params.language != "en") {
        fprintf(stderr, "error: language must be 'bg' or 'en'\n");
        return 1;
    }

    fprintf(stderr, "Loading model: %s\n", params.model.c_str());
    struct whisper_context_params context_params = whisper_context_default_params();
    context_params.use_gpu    = params.use_gpu;
    context_params.flash_attn = params.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), context_params);
    if (!ctx) {
        fprintf(stderr, "error: failed to initialize Whisper\n");
        return 2;
    }

    if (!whisper_is_multilingual(ctx) && params.language != "en") {
        fprintf(stderr, "error: the selected model is not multilingual\n");
        whisper_free(ctx);
        return 2;
    }

    audio_async audio(params.max_record_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        whisper_free(ctx);
        return 3;
    }

    std::ofstream output;
    if (!params.output.empty()) {
        output.open(params.output, std::ios::app);
        if (!output) {
            fprintf(stderr, "error: cannot open transcript file: %s\n", params.output.c_str());
            whisper_free(ctx);
            return 4;
        }
    }

    terminal_raw_mode terminal;
    if (!terminal.enable()) {
        fprintf(stderr, "error: terminal does not support interactive input\n");
        whisper_free(ctx);
        return 5;
    }

    bool recording = false;
    std::string last_transcription;
    auto started_at = std::chrono::steady_clock::now();

    print_dashboard(params.language, last_transcription, "● READY", "\033[1;32m");

    while (true) {
        const char key = read_key(recording ? 100 : -1);

        if (!recording && (key == 'q' || key == 'Q')) {
            break;
        }

        if (!recording && (key == 'c' || key == 'C')) {
            if (last_transcription.empty()) {
                print_dashboard(params.language, last_transcription, "● READY", "\033[1;32m", "Nothing to copy yet");
            } else if (copy_to_clipboard(last_transcription)) {
                print_dashboard(params.language, last_transcription, "● READY", "\033[1;32m", "Copied to clipboard");
            } else {
                print_dashboard(params.language, last_transcription, "● READY", "\033[1;32m", "Clipboard copy failed");
            }
            continue;
        }

        if (!recording && (key == 'b' || key == 'B' ||
                           key == 'e' || key == 'E')) {
            params.language = key == 'b' || key == 'B' ? "bg" :
                              "en";
            print_dashboard(params.language, last_transcription, "● READY", "\033[1;32m",
                    std::string("Language: ") + language_name(params.language));
            continue;
        }

        if (!recording && key == ' ') {
            if (!audio.resume() || !audio.clear()) {
                fprintf(stderr, "error: could not start recording\n");
                break;
            }
            recording = true;
            started_at = std::chrono::steady_clock::now();
            print_dashboard(params.language, last_transcription, "● RECORDING", "\033[1;31m");
            fprintf(stderr, "  Recording 00:00 — press SPACE to transcribe");
            fflush(stderr);
            continue;
        }

        if (recording && key == ' ') {
            std::vector<float> samples;
            audio.get(params.max_record_ms, samples);
            audio.pause();
            recording = false;
            fprintf(stderr, "%s", clear_line());

            if (samples.size() < WHISPER_SAMPLE_RATE / 4) {
                print_dashboard(params.language, last_transcription, "● READY", "\033[1;32m", "Recording was too short");
                continue;
            }

            struct whisper_full_params inference = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            inference.n_threads        = params.n_threads;
            inference.language         = params.language.c_str();
            inference.no_context       = true;
            inference.no_timestamps    = true;
            inference.print_progress   = false;
            inference.print_realtime   = false;
            inference.print_timestamps = false;
            std::atomic<int> progress { 0 };
            std::atomic<bool> finished { false };
            int inference_result = -1;
            inference.progress_callback           = progress_callback;
            inference.progress_callback_user_data = &progress;

            const auto transcription_started = std::chrono::steady_clock::now();
            print_dashboard(params.language, last_transcription, "◐ TRANSCRIBING", "\033[1;33m");
            std::thread worker([&]() {
                inference_result = whisper_full(ctx, inference, samples.data(), samples.size());
                finished.store(true);
            });

            const char spinner[] = { '|', '/', '-', '\\' };
            int spinner_frame = 0;
            while (!finished.load()) {
                const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - transcription_started).count();
                const int current_progress = progress.load();
                if (current_progress > 0) {
                    fprintf(stderr, "%s  Transcribing %c  %3d%%  %.1fs", clear_line(),
                            spinner[spinner_frame++ % 4], current_progress, elapsed_ms / 1000.0);
                } else {
                    fprintf(stderr, "%s  Transcribing %c  %.1fs", clear_line(),
                            spinner[spinner_frame++ % 4], elapsed_ms / 1000.0);
                }
                fflush(stderr);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            worker.join();
            fprintf(stderr, "%s  Transcribing complete — 100%%\n", clear_line());

            if (inference_result != 0) {
                fprintf(stderr, "\nerror: transcription failed\n");
                continue;
            }
            std::string transcription;
            const int segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < segments; ++i) {
                transcription += whisper_full_get_segment_text(ctx, i);
            }

            const size_t first = transcription.find_first_not_of(" \t\r\n");
            if (first != std::string::npos) {
                transcription.erase(0, first);
            }
            const size_t last = transcription.find_last_not_of(" \t\r\n");
            if (last != std::string::npos) {
                transcription.erase(last + 1);
            }
            last_transcription = transcription;

            if (output) {
                output << transcription << "\n\n";
                output.flush();
            }
            print_dashboard(params.language, last_transcription, "● READY", "\033[1;32m", "Transcription saved");
            continue;
        }

        if (recording) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - started_at).count();
            fprintf(stderr, "%s  Recording %02lld:%02lld — press SPACE to transcribe", clear_line(),
                    (long long) elapsed / 60, (long long) elapsed % 60);
            fflush(stderr);

            if (elapsed * 1000 >= params.max_record_ms) {
                fprintf(stderr, "%s  Maximum recording length reached — press SPACE to transcribe", clear_line());
            }
        }
    }

    if (recording) {
        audio.pause();
    }
    fprintf(stderr, "%s\nBye.\n", clear_line());
    whisper_free(ctx);
    return 0;
}
