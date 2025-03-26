// MAIN_CPP

#include "asr.h"
#include "ctc.h"
#include "libre.h"

#include <bits/stdc++.h>

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


/*int main() {
    std::cout << "=== Program STARTED ===\n";
    ASRModel model;
    vector<string> audio_files = {"test_data/lama.wav", "test_data/tiktok.wav"};
    vector<string> texts = {"lama", "tiktok"};
    
    for (int epoch = 0; epoch < 10; epoch++) {
        cout << "Epoch " << epoch << endl;
        for (size_t i = 0; i < audio_files.size(); i++) {
            cout << "Processing " << audio_files[i] << endl;
            auto spectrogram = get_processed_audio(audio_files[i]);
            auto transcript = text_to_indices(texts[i]);
            model.train(spectrogram, transcript);
        }
    }
    model.save("bin/trained_model.bin");
    return 0;
}*/
int main() {
    ASRModel model;
    
    // Загрузка данных
    string dataset_path = "../data/LibriSpeech/test-clean"; // Путь к тренировочным данным

    cout << "Loading data from: " << dataset_path << endl;
    
    auto data = load_librispeech_data(dataset_path);
    if (data.empty()) {
        cerr << "ERROR: No data loaded! Check path: " << dataset_path << endl;
        return 1;
    }
    cout << "Loaded " << data.size() << " samples" << endl;
    
    // Перемешиваем данные
    random_device rd;
    mt19937 g(rd());
    shuffle(data.begin(), data.end(), g);
    
    // Ограничиваем размер для тестирования
    const size_t max_samples = 1000; // Увеличивайте по мере тестирования
    if (data.size() > max_samples) {
        data.resize(max_samples);
    }
    
    // Обучение
    const int epochs = 100;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int processed = 0;
        
        for (const auto& [audio_path, transcript] : data) {
            try {
                // Пропускаем слишком длинные транскрипции
                if (transcript.length() > 100) continue;
                
                auto spectrogram = get_processed_audio(audio_path);
                auto indices = text_to_indices(transcript);
                
                model.train(spectrogram, indices);
                
                processed++;
                if (processed % 100 == 0) {
                    cout << "Processed " << processed << "/" << data.size() 
                         << " samples in epoch " << epoch + 1 << endl;
                }
            } catch (const exception& e) {
                cerr << "Error processing " << audio_path << ": " << e.what() << endl;
            }
        }
        
        cout << "Epoch " << epoch + 1 << " completed. Processed " << processed << " samples." << endl;
        
        // Сохраняем модель после каждой эпохи
        model.save("bin/librispeech_model_epoch_" + to_string(epoch + 1) + ".bin");
    }
    
    return 0;
}

/*int main() {
    ASRModel model;
    
    vector<string> audio_files = {"hw.wav", "lama.wav", "tiktok.wav", "long.wav"};
    vector<string> texts = {"hello world", "lama", "tiktok", "the quick brown fox jumps over the lazy dog"};
    
    for (int epoch = 0; epoch < 100; epoch++) {
        for (size_t i = 0; i < audio_files.size(); i++) {
            auto spectrogram = get_processed_audio(audio_files[i]);
            auto transcript = text_to_indices(texts[i]);
            model.train(spectrogram, transcript);
        }
    }
    
    model.save("trained_model.bin");
}*/

/*int main() {
    ASRModel model;
    model.load("trained_model.bin");
    
    auto spectrogram = get_processed_audio("test.wav");
    string text = model.recognize(spectrogram);
    cout << "Recognized: " << text << endl;
}*/
