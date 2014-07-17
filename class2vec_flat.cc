#include<cmath>
#include<string>
#include<iostream>
#include<vector>
int vec_size;
int word_num;
int class_num;
int max_record_words;
int min_record_words;
int min_word_count = 20;
float **word_vec;
float **class_vec;
ifstream train_file;
ofstream word_vec_file;
ofstream class_vec_file;
map<string, int> word_table;
map<string, int> class_table;

float expprod(float *vec1, float *vec2) {
    float p = 0;
    for (int i = 0; i < vec_size; i++) {
        p += vec1[i] * vec2[i];
    }
    return exp(p);
}

void init_net() {
    word_vec = new float *[word_num];
    for (int i = 0; i < word_num; i++) {
        word_vec[i] = new float[vec_size];
    }
    class_vec = new float *[class_num];
    for (int i = 0; i < class_num; i++) {
        class_vec[i] = new float[vec_size];
    }
}
void clear_vec(float *vec) {
    for (int i = 0; i < vec_size; i++) {
        vec[i] = 0;
    }
}

void update_vec(float *vec1, float *vec2, float *rate) {
    for (int i = 0; i < vec_size; i++) {
        vec1[i] += rate * vec2[i];
    }
}

int read_record(vector<string> &record) {
    string line;
    record.clear();
    if(!getline(train_file, line)) return 0;
    int begin = 0, end = 0;
    while(begin < line.length()) {
        while (begin < line.length() && line[begin] == '\t' || line[begin] == ' ') begin++;
        if (begin == line.length()) break;
        end = begin + 1;
        while(end < line.length() && end != '\t' && end != ' ') {
            end++;
        }
        record.push_back(line.substr(begin, end - begin));
        begin = end + 1;
    }
    return 1;
}

void learn_vocab() {
    train_file.seekg(0, train_file.beg);
    vector<string> record;
    class_num = 0;
    word_num = 0;
    map<string, int> word_count;
    while (read_record(record)) {
        if (record.size() == 0) continue;
        if (class_table.cound(record[0]) == 0) {
            class_table[record[0]] = class_num++;
        }
        for (int i = 1; i < record.size(); i++) {
            word_count[record[i]]++;
        }
    }
    for (map<string, int>::iterator it = word_count.begin(); it != word_count.end(); it++) {
        if (it->second < min_word_count) continue;
        word_table[it->first] = word_num++;
    }
}

int get_instance(int &class_id, vector<int> &words) {
    vector<string> record;
    words.clear();
    int nword = 0;
    if (read_record(record)) return 0;
    class_id = word_table[record[0]];
    for (int i = 1; i < record.size() && nword < max_record_words; i++) {
        if (word_table.count(record[i]) == 0) continue;
        nword++;
        words.append(word_table[record[i]]);
    }
    return 1;
}

void train() {
    int round = 0;
    vector<vector<float> > expprod_table(max_record_words, vector<float>(class_num, 0));
    float *vec = new float[vec_size];
    while (round < max_round) {
        int class_id;
        vector<int> words;
        train_file.seekg(0, train_file.beg);

        if (get_instance(class_id, words) == 0) {
            round++;
            continue;
        }

        if (words.size() < min_record_words) continue;

        float alpha = start_alpha * (1 - round / 100000);
        alpha = alpha < 0.0001 ? 0.0001 : alpha;

        float A = 0;
        for (int w = 0; w < words.size(); i++) {
            for (int c = 0; c < class_num; c++) {
                float a = expprod(word_vec[words[w]], class_vec[c]);
                expprod_table[w][c] = a;
                A += a;
            }
        }

        float s = 0;
        for (int w = 0; w < words.size(); w++) {
            s += expprod_table[w][class_id];
        }

        for (int w = 0; w < words.size(); w++) {
            clear_vec(vec);
            update(vec, class_vec[class_id], A * expprod_table[w][class_id]);
            for (int c = 0; c < class_num; c++) {
                update(vec, class_vec[c], - s * expprod_table[w][c]);
            }
            update(word_vec[w], vec, alpha / A / A);
        }

        for (int c = 0; c < class_num; c++) {
            clear_vec(vec);
            if (c == class_id) {
                for (w = 0; w < words.size(); w++) {
                    update(vec, word_vec[w], A * expprod_table[w][c]); 
                }
            }
            for (w = 0; w < words.size(); w++) {
                update(vec, words[w], - A * expprod_table[w][c]);
            }
            update(class_vec[c], vec, alpha / A / A);
        }
    }
}
void save_model() {
    for (map<string, int>::iterator it = word_table.begin(); it != word_table.end(); it++) {
        word_vec_file << it->first;
        for (int i = 0; i < vec_size; i++) {
            word_vec_file << ' ' << word_vec[it->second][i];
        }
        word_vec_file << endl;
    }
    for (map<string, in>::iterator it = class_table.begin(); it != class_table.end(); it++) {
        class_vec_file << it->first;
        for (int i = 0; i < vec_size; i++) {
            class_vec_file << ' ' << class_vec[it->second][i];
        }
        class_vec_file << endl;
    }
}

int main (int argc, char *argv[]) {
    train_file.open("train.dat");
    word_vec_file.open("word_vec.dat", ofstream::out);
    class_vec_file.open("class_vec.dat", ofstream::out);
    
    if (!(train_file.is_open() && word_vec_file.is_open() && class_vec_file.is_open())) {
        cerr << "CAN'T OPEN FILE." << endl;
        return -1;
    }
    learn_vocab();
    init_net();
    train();
    save_model();
    train_file.close();
    word_vec_file.close();
    class_vec_file.close();
    return 0;
}
