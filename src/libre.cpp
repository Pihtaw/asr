// LIBRE_CPP

#include "../include/libre.h"

#include <dirent.h>
#include <fstream>
#include <algorithm>

std::vector<std::pair<std::string, std::string>> load_librispeech_data(const std::string& base_path) {
    std::vector<std::pair<std::string, std::string>> data;
    
    auto process_chapter = [&](const std::string& speaker_path, const std::string& chapter_dir) {
        std::string chapter_path = speaker_path + "/" + chapter_dir;
        
        std::string speaker_id = speaker_path.substr(speaker_path.find_last_of('/') + 1);
        std::string trans_file = chapter_path + "/" + speaker_id + "-" + chapter_dir + ".trans.txt";
        
        std::ifstream trans(trans_file);
        if (!trans.is_open()) {
            std::cerr << "Cannot open transcript: " << trans_file << std::endl;
            return;
        }

        std::string line;
        while (std::getline(trans, line)) {
            size_t space_pos = line.find(' ');
            if (space_pos == std::string::npos) continue;
            
            std::string file_id = line.substr(0, space_pos);
            std::string transcript = line.substr(space_pos + 1);
            
            std::transform(transcript.begin(), transcript.end(), transcript.begin(), 
                         [](unsigned char c){ return std::tolower(c); });
            
            std::string audio_path = chapter_path + "/" + file_id + ".flac";
            data.emplace_back(audio_path, transcript);
        }
    };

    DIR* main_dir = opendir(base_path.c_str());
    if (!main_dir) {
        std::cerr << "Cannot open base directory: " << base_path << std::endl;
        return data;
    }

    struct dirent* entry;
    while ((entry = readdir(main_dir)) != nullptr) {
        std::string speaker_dir = entry->d_name;
        if (speaker_dir == "." || speaker_dir == "..") continue;
        
        std::string speaker_path = base_path + "/" + speaker_dir;
        DIR* speaker_dp = opendir(speaker_path.c_str());
        if (!speaker_dp) continue;

        struct dirent* chapter_entry;
        while ((chapter_entry = readdir(speaker_dp)) != nullptr) {
            std::string chapter_dir = chapter_entry->d_name;
            if (chapter_dir == "." || chapter_dir == "..") continue;
            
            process_chapter(speaker_path, chapter_dir);
        }
        closedir(speaker_dp);
    }
    closedir(main_dir);

    std::cout << "Loaded " << data.size() << " audio samples" << std::endl;
    return data;
}