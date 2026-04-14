#include <ostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

namespace py = pybind11;

struct pair_hash {
    inline size_t operator()(const std::pair<int, int> & v) const {
        return v.first * 31 + v.second;
    }
};

class Tokenizer {
    std::unordered_map<std::pair<int, int>, int, pair_hash> mergeForest;
    std::unordered_map<int, std::vector<uint8_t>> vocab;

    std::vector<int> getCodePoints(const std::string& data);
    std::unordered_map<std::pair<int, int>, int, pair_hash> getPairs(const std::vector<int>& tokens);
    std::vector<int> doMerge(const std::vector<int>& tokens, int size, const std::pair<int, int>& pair, int idx);

public:
    Tokenizer();
    ~Tokenizer() = default;

    void train(const std::string& data, int nVocab);
    std::vector<int> encode(const std::string& text);
    py::bytes decode(const std::vector<int>& tokens);

    void displayParams() {
        std::cout << "Vocab:" << std::endl;
        for (const auto& kv : vocab) {
            std::cout << kv.first << " : (";
            for (const auto& val : kv.second) {
                std::cout << val << ", ";
            }
            std::cout << ")" << std::endl;
        }
    }
};

PYBIND11_MODULE(scribe, m) {
    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("train", &Tokenizer::train)
        .def("encode", &Tokenizer::encode)
        .def("decode", &Tokenizer::decode)
        .def("displayParams", &Tokenizer::displayParams);
}