#include "ocria_latin.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model_dir> <image_path> [options]\n"
              << "  --long-side <N>   Detector max long side (default 1600)\n"
              << "  --threads <N>     ORT intra-op threads (default 0 = auto)\n"
              << "  --repeat <N>      Run N times and report warm timings (default 1)\n"
              << "  --json            Emit JSON output\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    std::string model_dir = argv[1];
    std::string image_path = argv[2];
    ocria::OcriaConfig cfg;
    int repeat = 1;
    bool as_json = false;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--long-side" && i + 1 < argc) {
            cfg.det_long_side = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            cfg.threads = std::atoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--json") {
            as_json = true;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    int w = 0, h = 0, ch = 0;
    unsigned char* pixels = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
    if (!pixels) {
        std::cerr << "Failed to load image: " << image_path << " (" << stbi_failure_reason() << ")\n";
        return 2;
    }

    try {
        ocria::OcriaLatinEngine engine(model_dir, cfg);
        std::vector<ocria::OcrLine> lines;
        ocria::OcrTimings timings;
        for (int r = 0; r < repeat; ++r) {
            lines = engine.run(pixels, w, h, 3, &timings);
        }
        stbi_image_free(pixels);

        if (as_json) {
            std::cout << "{\n  \"image\": \"" << json_escape(image_path) << "\",\n"
                      << "  \"width\": " << w << ", \"height\": " << h << ",\n"
                      << "  \"timings_ms\": {\"det\": " << timings.det_ms
                      << ", \"crop\": " << timings.crop_ms
                      << ", \"rec\": " << timings.rec_ms
                      << ", \"total\": " << timings.total_ms << "},\n"
                      << "  \"lines\": [\n";
            for (size_t i = 0; i < lines.size(); ++i) {
                const auto& l = lines[i];
                std::cout << "    {\"text\": \"" << json_escape(l.text) << "\", "
                          << "\"confidence\": " << l.confidence << ", "
                          << "\"bbox\": [" << l.bbox[0] << ", " << l.bbox[1] << ", "
                          << l.bbox[2] << ", " << l.bbox[3] << "]}"
                          << (i + 1 < lines.size() ? "," : "") << "\n";
            }
            std::cout << "  ]\n}\n";
        } else {
            std::fprintf(stderr,
                         "[ocria_cli] %dx%d | lines=%zu | det=%.1fms crop=%.1fms rec=%.1fms total=%.1fms\n",
                         w, h, lines.size(),
                         timings.det_ms, timings.crop_ms, timings.rec_ms, timings.total_ms);
            for (const auto& l : lines) {
                std::printf("[%.2f] (%4.0f,%4.0f,%4.0f,%4.0f) %s\n",
                            l.confidence, l.bbox[0], l.bbox[1], l.bbox[2], l.bbox[3],
                            l.text.c_str());
            }
        }
    } catch (const std::exception& e) {
        stbi_image_free(pixels);
        std::cerr << "Error: " << e.what() << "\n";
        return 3;
    }
    return 0;
}
