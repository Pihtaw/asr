#include "asr.h"
#include "ctc.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

vector<int> text_to_indices(const string& text) {
    const string& alphabet = "abcdefghijklmnopqrstuvwxyz' ";
    vector<int> indices;
    for (char c : text) {
        char lower_c = tolower(c);
        size_t pos = alphabet.find(lower_c);
        if (pos == string::npos) {
            if (c == ' ') pos = alphabet.find(' ');
            if (pos == string::npos) {
                cerr << "Error: Character '" << c << "' not in alphabet\n";
                throw runtime_error("Invalid character in transcript");
            }
        }
        indices.push_back(static_cast<int>(pos));
    }
    return indices;
}

int main() {
    try {
        ASRModel model;
        string audio_path = "input0.wav";
        
        // Проверка файла
        ifstream file(audio_path);
        if (!file) throw runtime_error("Cannot open audio file: " + audio_path);
        file.close();

        // Загрузка и проверка аудио
        auto spectrogram = get_processed_audio(audio_path);
        if (spectrogram.empty()) throw runtime_error("Empty spectrogram");
        
        // Транскрипция
        auto transcript = text_to_indices("hello world");
        cout << "Transcript indices: ";
        for (int idx : transcript) cout << idx << " ";
        cout << endl;

        // Обучение (увеличьте эпохи при необходимости)
        for (int epoch = 0; epoch < 150; ++epoch) {
            model.train(spectrogram, transcript);
            if (epoch % 5 == 0) cout << "Epoch " << epoch << endl;
        }

        // Распознавание
        string text = model.recognize(spectrogram);
        cout << "Recognized: " << text << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}