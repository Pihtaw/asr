#ifndef LIBRE_H
#define LIBRE_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype> 
#include <sys/types.h> 
#include <dirent.h>
#include <sndfile.h> 

std::vector<std::pair<std::string, std::string>> load_librispeech_data(const std::string& base_path);

#endif // LIBRE_H