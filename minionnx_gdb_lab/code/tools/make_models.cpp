// =====================================================================================
// tools/make_models.cpp  --  write the example .onnx files into models/
// =====================================================================================
//   make_models [output_dir]          (default: models)
//
// Each file shows a different way of STORING WEIGHTS (Stage 1):
//   xor.onnx                     float_data   (a packed list of floats inside the protobuf)
//   classifier.onnx              raw_data     (raw little-endian bytes; what PyTorch writes)
//   branchy.onnx                 raw_data, 4 parallel branches (for the parallel executor)
//   branchy_external.onnx(+.data) external data: weights in a separate, 64-byte-aligned file
// =====================================================================================
#include "model_zoo.hpp"

#include <cstdio>

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "models";
    try {
        mo::SaveOptions float_data;
        float_data.use_raw_data = false;
        mo::save_model_file(zoo::make_xor(), dir + "/xor.onnx", float_data);
        std::printf("wrote %s/xor.onnx (weights as float_data)\n", dir.c_str());

        mo::save_model_file(zoo::make_classifier(16, 32, 5), dir + "/classifier.onnx");
        std::printf("wrote %s/classifier.onnx (weights as raw_data, Gemm transB=1 like PyTorch)\n", dir.c_str());

        mo::save_model_file(zoo::make_branchy(4, 32, 64, 7), dir + "/branchy.onnx");
        std::printf("wrote %s/branchy.onnx (4 independent branches)\n", dir.c_str());

        mo::SaveOptions ext;
        ext.external_data = true;
        mo::save_model_file(zoo::make_branchy(4, 32, 64, 7), dir + "/branchy_external.onnx", ext);
        std::printf("wrote %s/branchy_external.onnx + .data (weights in a separate file)\n", dir.c_str());
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n(does the folder '%s' exist? run from the project root)\n", e.what(), dir.c_str());
        return 1;
    }
    return 0;
}
