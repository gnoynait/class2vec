#include<cstdlib>
#include<cmath>
#include<cassert>
#include<cstring>
#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include<map>
using namespace std;

// dimension of vector
int vec_size;
// vocabulary size
int word_num;
// number of class
int class_num;
int max_record_words = 10;
int min_record_words = 3;
int min_word_count = 20;

float **word_vec;
float **class_vec;

map<string, int> word_table;
vector<string> class_table;

ifstream word_vec_file;
ifstream class_vec_file;
ifstream target_file;
ofstream result_file;

// exp(vec1 * vec2)
float expprod(float *vec1, float *vec2) {
    float p = 0;
    for (int i = 0; i < vec_size; i++) {
        p += vec1[i] * vec2[i];
    }
    return exp(p);
}

// allocate memory for word_vec and class_vec
void init_net() {
    word_vec = new float *[word_num];
    class_vec = new float *[class_num];
    if (!(word_vec && class_vec)) {
        cerr << "MEMOERY ALLOCATION ERROR" << endl;
        exit(-1);
    }
    for (int i = 0; i < word_num; i++) {
        word_vec[i] = new float[vec_size];
        assert(word_vec[i]);
    }
    for (int i = 0; i < class_num; i++) {
        class_vec[i] = new float[vec_size];
        assert(class_vec[i]);
    }
}

void load_model() {
    string word, class_name;

    word_vec_file >> word_num >> vec_size;
    class_vec_file >> class_num;

    init_net();
    for (int w = 0; w < word_num; w++) {
        word_vec_file >> word;
        word_table[word] = w;
        for (int j = 0; j < vec_size; j++) {
            word_vec_file >> word_vec[w][j];
        }
    }

    for (int c = 0; c < class_num; c++) {
        class_vec_file >> class_name;
        class_table.push_back(class_name);
        for (int j = 0; j < vec_size; j++) {
            class_vec_file >> class_vec[c][j];
        }
    }
}

int read_record(vector<string> &record) {
    string line;
    record.clear();
    if(!getline(target_file, line)) return 0;

    int begin = 0, end;
    while(begin < line.length()) {
        while (begin < line.length() && (line[begin] == '\t'
            || line[begin] == ' ')) begin++;
        if (begin == line.length()) break;
 
        end = begin + 1;
        while(end < line.length() && line[end] != '\t' && line[end] != ' ') {
            end++;
        }
        record.push_back(line.substr(begin, end - begin));
        begin = end + 1;
    }
    return 1;
}

int get_instance(vector<int> &words) {
    vector<string> record;
    words.clear();
    int nword = 0;
    if (read_record(record) == 0) return 0;
    for (int i = 0; i < record.size() && nword < max_record_words; i++) {
        if (word_table.count(record[i]) == 0) continue;
        nword++;
        words.push_back(word_table[record[i]]);
    }
    return 1;
}

float product(float *vec1, float *vec2) {
    float p = 0.0;
    for (int i = 0; i < vec_size; i++) {
        p += vec1[i] * vec2[i];
    }
    return p;
}

int predict(vector<int> &words) {
    float maxv = -10000;
    int result = -1;
    for (int c = 0; c < class_num; c++) {
        float v = 0;
        
        for (int w = 0; w < words.size(); w++) {
            v += product(word_vec[words[w]], class_vec[c]);
        }
        if (v > maxv) {
            result = c;
            maxv = v;
        }
    }
    printf("%d\n", result);
    return result;
}

void process() {
    vector<int> words;
    while(get_instance(words)) {
        if (words.size() < min_record_words) {
            result_file << "IGNORE" << endl;
            continue;
        }
        int result = predict(words);
        result_file << class_table[result] << endl;
    }
}

int main (int argc, char *argv[]) {
    word_vec_file.open("word_vec.dat");
    class_vec_file.open("class_vec.dat");
    target_file.open("target.dat");
    result_file.open("result.dat", ofstream::out);

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-w") == 0) {
            max_record_words = atoi(argv[i + 1]);
        }
        if (strcmp(argv[i], "-n") == 0) {
            min_record_words = atoi(argv[i + 1]);
        }
    }

    if (!(target_file.is_open() && result_file.is_open()
        && word_vec_file.is_open() && class_vec_file.is_open())) {
        cerr << "CAN'T OPEN FILE." << endl;
        return -1;
    }

    load_model();
    process();

    word_vec_file.close();
    class_vec_file.close();
    target_file.close();
    result_file.close();

    return 0;
}
