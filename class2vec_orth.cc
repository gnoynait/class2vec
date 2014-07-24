#include<cmath>
#include<cstdio>
#include<cassert>
#include<map>
#include<ctime>
#include<cstdlib>
#include<cstring>
#include<string>
#include<iostream>
#include<fstream>
#include<vector>
using namespace std;

// dimension of word and class vector
int vec_size = 20;
// size of vocabulary
int word_num;
// number of class
int class_num;
// maximum words in a record, remaining words will be ignored
int max_record_words = 100;
// minimum words in a record, record has fewer words will be ignored
int min_record_words = 1;
// minimum apperance of a word, words appeared fewer than this will be ignored
int min_word_count = 20;
// how many times to go through th train file
int max_round = 2;
// initial update_vec rate
float alpha = 0.1;
float lambda = 1.0;

float **word_vec;
float **class_vec;

// index in word_vec of a word
map<string, int> word_table;
// index in class_vec of a class
map<string, int> class_table;

ifstream train_file;
ofstream word_vec_file;
ofstream class_vec_file;

// exp(vec1 * vec2)
float expprod(float *vec1, float *vec2) {
    float p = 0;
    for (int i = 0; i < vec_size; i++) {
        p += vec1[i] * vec2[i];
    }
    return exp(p);
}

float product(float *vec1, float *vec2) {
    float p = 0.0;
    for (int i = 0; i < vec_size; i++) {
        p += vec1[i] * vec2[i];
    }
    return p;
}
// set vec = 0
void clear_vec(float *vec) {
    for (int i = 0; i < vec_size; i++) {
        vec[i] = 0;
    }
}

// set vec1 = vec1 + rate * vec2
void update_vec(float *vec1, float *vec2, float rate) {
    for (int i = 0; i < vec_size; i++) {
        vec1[i] += rate * vec2[i];
		// TODO delete this
		/*
		if (isnan(vec1[i])) {
			printf ("nan\n");
			getchar();
		}
		*/
    }
}

void normalize(float *vec) {
    float s = 0;
    for (int i = 0; i < vec_size; i++) {
        s += vec[i] * vec[i];
    }
    s = sqrt(s);
    if (s == 0) return;
    for (int i = 0; i < vec_size; i++) {
        vec[i] /= s;
    }
}


// allocate memory for word_vec and class_vec and give 
// them random initial value
void init_net() {
    word_vec = new float *[word_num];
    class_vec = new float *[class_num];
    if (!(word_vec && class_vec)) {
        cerr << "MEMORY ALLOCATION ERROR" << endl;
        exit(-1);
    }

    srand(time(NULL));
    for (int i = 0; i < word_num; i++) {
        word_vec[i] = new float[vec_size];
        assert(word_vec[i]);
        for (int j = 0; j < vec_size; j++) {
            word_vec[i][j] = 2.0 * ((float) rand() / RAND_MAX - 0.5);
        }
        normalize(word_vec[i]);
    }
    for (int i = 0; i < class_num; i++) {
        class_vec[i] = new float[vec_size];
        assert(class_vec[i]);
        for (int j = 0; j < vec_size; j++) {
            class_vec[i][j] = 2.0 * ((float) rand() / RAND_MAX - 0.5);
        }
        normalize(class_vec[i]);
    }
}

// read a record, which is a list of string, from train_file
// return 1 if success else 0
int read_record(vector<string> &record) {
    string line;
    record.clear();
    if(!getline(train_file, line)) return 0;

    int begin = 0, end = 0;
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

// learn vocabulary and classes from the train file
void learn_vocab() {
    train_file.seekg(0, ios::beg);
    vector<string> record;
    class_num = 0;
    word_num = 0;
    map<string, int> word_count;
    while (read_record(record)) {
        if (record.size() == 0) continue;
        if (class_table.count(record[0]) == 0) {
            class_table[record[0]] = class_num++;
        }
        for (int i = 1; i < record.size(); i++) {
            word_count[record[i]]++;
        }
    }
    for (map<string, int>::iterator it = word_count.begin();
        it != word_count.end(); it++) {
        if (it->second < min_word_count) continue;
        word_table[it->first] = word_num++;
    }
}

// get a instance form train file, words are transformed to its index
// return 1 if success else 0
int get_instance(int &class_id, vector<int> &words) {
    vector<string> record;
    words.clear();
    int nword = 0;
    if (read_record(record) == 0) return 0;
    class_id = word_table[record[0]];
    for (int i = 1; i < record.size() && nword < max_record_words; i++) {
        if (word_table.count(record[i]) == 0) continue;
        nword++;
        words.push_back(word_table[record[i]]);
    }
    return 1;
}

// train the model max_round times
void train() {
    int round = 0;
    int record_count = 0;
    int round_records = 0;
	int class_id;
	vector<int> words;
	// ws = Sum(v_w)
	float *ws = new float[vec_size];
	float *deta_w = new float[vec_size];
	float *deta_c = new float[vec_size];

    train_file.clear();
    train_file.seekg(0, ios::beg);
    cerr << "cur " << train_file.tellg() << endl;
    while (round < max_round) {
        if (get_instance(class_id, words) == 0) {
            round++;
            cerr << round << ' ' << round_records << ' ' << alpha << endl;
            round_records = 0;
            train_file.clear();
            train_file.seekg(0, ios::beg);
            alpha = alpha * 0.8;
            alpha = alpha > 0.0001 ? alpha : 0.0001;
            continue;
        }
        round_records++;

        if (words.size() < min_record_words) continue;

        if (record_count < 100000) {
            record_count++;
        }

		clear_vec(ws);
		for (int i = 0; i < words.size(); i++) {
			int w = words[i];
			update_vec(ws, word_vec[w], 1);
		}

		clear_vec(deta_w);
		for (int c = 0; c < class_num; c++) {
            float p = product(ws, class_vec[c]);
            if (c == class_id) {
                update_vec(deta_w, class_vec[c], 1 - p);
            } else {
                update_vec(deta_w, class_vec[c], - p);
            }
		}

		for (int i = 0; i < words.size(); i++) {
			int w = words[i];
			update_vec(word_vec[w], deta_w, alpha);
		}

		for (int c = 0; c < class_num; c++) {
			clear_vec(deta_c);
            float p = product(class_vec[c], ws);
			if (c == class_id) {
                update_vec(class_vec[c], ws, alpha * lambda * (1 - p));
                normalize(class_vec[c]);
			} else {
                //update_vec(class_vec[c], ws, - alpha * p);
            }
		}
    }
}

// save word_vec and class_vec into file
void save_model() {
    word_vec_file << word_num << '\t' << vec_size << endl;
    for (map<string, int>::iterator it = word_table.begin();
        it != word_table.end(); it++) {
        word_vec_file << it->first;
        for (int i = 0; i < vec_size; i++) {
            word_vec_file << '\t' << word_vec[it->second][i];
        }
        word_vec_file << endl;
    }

    class_vec_file << class_num << endl;
    for (map<string, int>::iterator it = class_table.begin();
        it != class_table.end(); it++) {
        class_vec_file << it->first;
        for (int i = 0; i < vec_size; i++) {
            class_vec_file << '\t' << class_vec[it->second][i];
        }
        class_vec_file << endl;
    }
}

int main (int argc, char *argv[]) {
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0) {
            vec_size = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-m") == 0) {
            min_word_count = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-r") == 0) {
            max_round = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-n") == 0) {
            min_record_words = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-w") == 0) {
            max_record_words = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-l") == 0) {
            lambda = atof(argv[ i + 1]);
        }
    }

    train_file.open("train.dat");
    word_vec_file.open("word_vec.dat", ofstream::out);
    class_vec_file.open("class_vec.dat", ofstream::out);
    if (!(train_file.is_open() && word_vec_file.is_open()
        && class_vec_file.is_open())) {
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
