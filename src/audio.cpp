// AUDIO_CPP

#include "../include/asr.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <complex>
#include <sndfile.h>


std::vector<float> read_flac(const std::string& filename) {
    SF_INFO sfinfo;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!file) {
        std::cerr << "Error opening FLAC file: " << filename << std::endl;
        return {};
    }
    
    std::vector<float> audio_data(sfinfo.frames * sfinfo.channels);
    sf_read_float(file, audio_data.data(), audio_data.size());
    sf_close(file);
    
    // Если аудио стерео, усредняем до моно
    if (sfinfo.channels > 1) {
        std::vector<float> mono_data(sfinfo.frames);
        for (int i = 0; i < sfinfo.frames; i++) {
            float sum = 0.0f;
            for (int c = 0; c < sfinfo.channels; c++) {
                sum += audio_data[i * sfinfo.channels + c];
            }
            mono_data[i] = sum / sfinfo.channels;
        }
        return mono_data;
    }
    
    return audio_data;
}

// Чтение WAV файла
std::vector<float> read_wav(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }
    file.seekg(44);
    
    std::vector<float> audio_data;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        audio_data.push_back(sample / 32768.0f);
    }
    return audio_data;
}

// Реализация FFT
void fft(std::vector<std::complex<float>>& x) {
    const size_t N = x.size();
    if (N <= 1) return;

    std::vector<std::complex<float>> even(N/2);
    std::vector<std::complex<float>> odd(N/2);
    for (size_t i = 0; i < N/2; ++i) {
        even[i] = x[2*i];
        odd[i] = x[2*i + 1];
    }

    fft(even);
    fft(odd);

    for (size_t k = 0; k < N/2; ++k) {
        float angle = -2.0f * static_cast<float>(M_PI) * k / N;
        std::complex<float> t = std::polar(1.0f, angle) * odd[k];
        x[k] = even[k] + t;
        x[k + N/2] = even[k] - t;
    }
}

// Вычисление спектра мощности
std::vector<float> compute_fft(const std::vector<float>& frame) {
    size_t N = frame.size();
    std::vector<std::complex<float>> complex_frame(N);

    for (size_t i = 0; i < N; ++i) {
        float window = 0.54f - 0.46f * cos(2 * M_PI * i / (N - 1));
        complex_frame[i] = frame[i] * window;
    }

    fft(complex_frame);

    std::vector<float> spectrum(N/2 + 1);
    for (size_t i = 0; i <= N/2; ++i) {
        spectrum[i] = std::abs(complex_frame[i]);
    }

    return spectrum;
}
// Преобразование частоты в мел-шкалу
float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

// Преобразование мел-шкалы в частоту
float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Создание банка мел-фильтров
std::vector<std::vector<float>> create_mel_filter_bank() {
    float min_mel = hz_to_mel(0);
    float max_mel = hz_to_mel(SAMPLE_RATE / 2.0f);
    
    // Равномерно распределенные точки на мел-шкале
    std::vector<float> mel_points(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; ++i) {
        mel_points[i] = min_mel + i * (max_mel - min_mel) / (N_MELS + 1);
    }
    
    // Преобразуем обратно в герцы
    std::vector<float> hz_points(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    // Преобразуем в индексы FFT
    std::vector<int> bin(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; ++i) {
        bin[i] = floor((FRAME_SIZE / 2 + 1) * hz_points[i] / (SAMPLE_RATE / 2.0f));
    }
    
    // Создаем треугольные фильтры
    std::vector<std::vector<float>> filter_bank(N_MELS, std::vector<float>(FRAME_SIZE/2 + 1, 0.0f));
    
    for (int i = 0; i < N_MELS; ++i) {
        int start = bin[i], center = bin[i+1], end = bin[i+2];
        
        // Возрастающая часть фильтра
        for (int j = start; j < center; ++j) {
            filter_bank[i][j] = (j - start) / float(center - start);
        }
        
        // Убывающая часть фильтра
        for (int j = center; j < end; ++j) {
            filter_bank[i][j] = (end - j) / float(end - center);
        }
    }
    
    return filter_bank;
}

// Применение мел-фильтров
std::vector<float> apply_mel_filters(const std::vector<float>& spectrum, 
                                   const std::vector<std::vector<float>>& filter_bank) {
    std::vector<float> mel_energies(N_MELS, 0.0f);
    
    for (int i = 0; i < N_MELS; ++i) {
        for (size_t j = 0; j < spectrum.size(); ++j) {
            mel_energies[i] += spectrum[j] * filter_bank[i][j];
        }
        // Логарифмирование энергии (добавляем небольшое значение для стабильности)
        mel_energies[i] = logf(mel_energies[i] + 1e-6f);
    }
    
    return mel_energies;
}

// Сохранение данных для визуализации
void save_plot_data(const std::vector<float>& data, const std::string& filename) {
    std::ofstream file(filename);
    for (size_t i = 0; i < data.size(); ++i) {
        file << i << " " << data[i] << "\n";
    }
}

// Основная функция обработки аудио
std::vector<std::vector<float>> compute_mel_spectrogram(const std::vector<float>& audio) {
    static auto mel_filters = create_mel_filter_bank();
    std::vector<std::vector<float>> spectrogram;
    
    for (size_t i = 0; i + FRAME_SIZE <= audio.size(); i += HOP_LENGTH) {
        // Вырезаем фрейм
        std::vector<float> frame(audio.begin() + i, audio.begin() + i + FRAME_SIZE);
        
        // Вычисляем FFT
        auto spectrum = compute_fft(frame);
        
        // Сохраняем спектр первого фрейма для визуализации
        if (i == 0) {
            save_plot_data(spectrum, "spectrum.dat");
        }
        
        // Применяем мел-фильтры
        auto mel_energies = apply_mel_filters(spectrum, mel_filters);
        spectrogram.push_back(mel_energies);
        
        // Сохраняем мел-спектр первого фрейма
        if (i == 0) {
            save_plot_data(mel_energies, "mel_energies.dat");
        }
    }
    
    return spectrogram;
}

// Новая функция для получения готовых данных
std::vector<std::vector<std::vector<float>>> get_processed_audio(const std::string& filename) {
    auto audio = read_wav(filename);
    auto spectrogram = compute_mel_spectrogram(audio);
    
    // Преобразуем в 3D тензор [1, N_MELS, time_frames]
    std::vector<std::vector<std::vector<float>>> input(1);
    for (const auto& frame : spectrogram) {
        input[0].push_back(frame);
    }
    
    // Транспонируем [1, time_frames, N_MELS]
    std::vector<std::vector<std::vector<float>>> transposed(1, 
        std::vector<std::vector<float>>(
            input[0][0].size(),
            std::vector<float>(input[0].size())
        ));
    
    for (size_t i = 0; i < input[0].size(); ++i) {
        for (size_t j = 0; j < input[0][i].size(); ++j) {
            transposed[0][j][i] = input[0][i][j];
        }
    }
    
    return transposed;
}

/*int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.wav>" << std::endl;
        return 1;
    }
    
    // 1. Чтение аудиофайла
    auto audio = read_wav(argv[1]);
    if (audio.empty()) {
        return 1;
    }
    
    // 2. Вычисление мел-спектрограммы
    auto spectrogram = compute_mel_spectrogram(audio);
    
    // 3. Сохранение результатов
    std::ofstream out("spectrogram.dat");
    for (const auto& frame : spectrogram) {
        for (float val : frame) {
            out << val << " ";
        }
        out << "\n";
    }
    
    // 4. Генерация графиков с помощью gnuplot
    std::ofstream gp_script("plot.gp");
    gp_script << "set terminal png enhanced\n"
              << "set output 'spectrum.png'\n"
              << "set title 'FFT Spectrum (First Frame)'\n"
              << "set xlabel 'Frequency Bin'\n"
              << "set ylabel 'Magnitude'\n"
              << "plot 'spectrum.dat' with lines title 'Spectrum'\n"
              << "set output 'mel.png'\n"
              << "set title 'Mel Filter Bank Energies (First Frame)'\n"
              << "set xlabel 'Mel Filter Index'\n"
              << "set ylabel 'Log Energy'\n"
              << "plot 'mel_energies.dat' with lines title 'Mel Energies'\n";
    gp_script.close();
    
    system("gnuplot plot.gp");
    
    std::cout << "Processing complete. Results saved to:\n"
              << "- spectrogram.dat (full spectrogram)\n"
              << "- spectrum.png (FFT spectrum of first frame)\n"
              << "- mel.png (Mel energies of first frame)" << std::endl;
    
    return 0;
}*/
