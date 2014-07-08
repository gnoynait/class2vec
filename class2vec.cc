#include<string>
#include<cstring>
#include<iostream>
#include<fstream>
#include<cmath>
#include<cstdlib>
#include<vector>
#include<map>
#include<set>
#include<cassert>
using namespace std;
#define MAX_CODE_LEN 60
#define MAX_RECORD_WORDS 60
#define MAX_WORD_LEN 60
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
// map code to its index in the syn1
map<string, int> code_index;
// parent_index[i] is the index in syn1 of parent of the node who's index in syn1 is i
int *parent_index;
int *children_index;
// map vocab word to its index in the syn0
map<string, int> vocab_index;
//int latest_code_index = 0;
// the size of each vector
int vec_size = 100;
// how many vocab word in the table
int vocab_size;
// how many class
int node_num;
// if a word appears less than min_word_count, it will be ignored
int min_word_count;
// syn0: vector table for word
// syn1: vector table for class
// neu:  vecotr for middle sum unit
// eneu: error vector for middle sum unit
float *syn0, *syn1, *neu, *eneu;
// for fast compute 1/(1 + exp(x))
float *expTable;
// inital parameter changing step
float starting_alpha = 0.025;
int max_round = 1000;
// read a word from fin
// return: 0 if '\n'
//         len if read a word
//        -1 if end of file
int read_word(FILE *fin, char *buffer) {
    int len = 0;
    char ch;
    if (feof(fin)) {
        return -1;
    }
    while ((ch = fgetc(fin)) != EOF) {
        if (ch == '\n') {
            if (len > 0) {
                ungetc(ch, fin);
            }
            break;
        } else if (ch == ' ' || ch == '\t') {
            if (len == 0) continue;
            break;
        } else if (len < MAX_WORD_LEN - 1) {
            buffer[len++] = ch;
        }
    }
    buffer[len] = '\0';
    if (len == 0 && ch == EOF) return -1;
    return len;
}

void learn_vocab(FILE *train_file) {
    char word[MAX_WORD_LEN];
    int word_len;
    map<string, int> word_count; // count each number for filter words
    set<string> codes;

    fseek(train_file, 0, SEEK_SET);
    int new_record = 1;
    while((word_len = read_word(train_file, word)) != -1) {
        if (word_len == 0) {
            new_record = 1;
        } else if (new_record) {
            codes.insert(word);
            new_record = 0;
        } else {
            word_count[word]++;
        }
    }

    int word_next_index = 0;
    for (map<string, int>::iterator it = word_count.begin(); it != word_count.end(); ++it) {
        if (it->second >= min_word_count) {
            vocab_index[it->first] = word_next_index++;
        }
    }
    vocab_size = vocab_index.size();

    int node_next_index = 1;
    node_num = codes.size() - 1;
    parent_index = new int[node_num]();
    children_index = new int[node_num * 2]();
    for (set<string>::iterator it = codes.begin(); it != codes.end(); ++it) {
        string code = *it;
        int pre = 0;
        for (int i = 0; i < code.length() - 1; ++i) {
            int c = code[i] - '0';
            assert(c == 0 || c == 1);
            if (children_index[pre * 2 + c] == 0) {
                children_index[pre * 2 + c] = node_next_index;
                parent_index[node_next_index] = pre;
                ++node_next_index;
            }
            pre = children_index[pre * 2 + c];
        }
    }
}

void init_net() {
    syn0 = new float[vec_size * vocab_size];
    syn1 = new float[vec_size * node_num]();
    neu = new float[vec_size];
    eneu = new float[vec_size];
    expTable = new float[EXP_TABLE_SIZE + 1];
    if (!(syn0 && syn1 && neu && eneu && expTable)) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    for (int i = 0; i < node_num; ++i) {
        for (int j = 0; j < vec_size; ++j) {
            syn1[i * vec_size + j] = (rand() / (float)RAND_MAX - 0.5) / vec_size;
        }
    }
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < vec_size; ++j) {
            syn0[i * vec_size + j] = (rand() / (float)RAND_MAX - 0.5) / vec_size;
        }
    }
}

int read_record(FILE *fin, int *code, int *nodes, int *words, int &len_code, int &len_words) {
    char buffer[MAX_WORD_LEN];
    len_code = 0;
    len_words = 0;
    int len;
    int read_code = 1;
    while ((len = read_word(fin, buffer)) != -1) {
        if (len == 0) {
            if (read_code == 0) break;
        } else if (read_code) {
            int node;
            int pre = 0;
            len = len <= MAX_CODE_LEN ? len : MAX_CODE_LEN;
            for (int i = 0; i < len; ++i) {
                assert(buffer[i] == '0' || buffer[i] == '1');
                code[i] = buffer[i] - '0';
                nodes[i] = pre;
                pre = children_index[pre * 2 + buffer[i] - '0'];
            }
            len_code = len;
            read_code = 0;
        } else if (vocab_index.count(buffer) > 0 && len_words < MAX_RECORD_WORDS) {
            words[len_words] = vocab_index[buffer];
            ++len_words;
        }
    }
    return len_code > 0;
}

void train_model(FILE *train_file) {
    int code[MAX_CODE_LEN];
    int nodes[MAX_CODE_LEN];
    int words[MAX_RECORD_WORDS];
    int len_code, len_words;
    int record_count = 0;
    float alpha;
    fseek(train_file, 0, SEEK_SET);
    while (read_record(train_file, code, nodes, words, len_code, len_words )) {
        //cerr << record_count << endl;
        alpha = starting_alpha * (1 - record_count / (float)100000);
        if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        for (int i = 0; i < vec_size; ++i) {
            neu[i] = 0;
            eneu[i] = 0;
        }
        for (int i = 0; i < len_words; ++i) {
            for (int j = 0; j < vec_size; ++j) {
                neu[j] += syn0[words[i] * vec_size + j];
            }
        }
        for (int i = 0; i < len_code; ++i) {
            float f = 0;
            float g;
            for (int j = 0; j < vec_size; ++j) {
                f += neu[j] * syn1[nodes[i] * vec_size + j];
            }
            if (f <= -MAX_EXP) continue;
            else if (f >= MAX_EXP) continue;
            else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            g = (1 - code[i] - f) * alpha;
            for (int c = 0; c < vec_size; ++c) {
                eneu[c] += g * syn1[nodes[i] * vec_size + c];
                syn1[nodes[i] * vec_size + c] += g * neu[c];
            }
        }
        for (int i = 0; i < len_words; i++) {
            for (int c = 0; c < vec_size; c++) {
                syn0[c + words[i] * vec_size] += eneu[c];
            }
        }
        ++record_count;
    }
}

void dfs_save_code(FILE *fout, int root, char *code, int len) {
    fprintf(fout, "/");
    for (int i = 0; i < len; ++i) {
        fprintf(fout, "%c", code[i]);
    }
    fprintf (fout, "\t");
    for (int i = 0; i < vec_size; ++i) {
        char sep = i == vec_size - 1 ? '\n' : '\t';
        fprintf(fout, "%f%c", syn1[root * vec_size + i], sep);
    }
    if (children_index[2 * root]) {
        code[len] = '0';
        dfs_save_code(fout, children_index[2 * root], code, len + 1);
    }
    if (children_index[2 * root + 1]) {
        code[len] = '1';
        dfs_save_code(fout, children_index[2 * root + 1], code, len + 1);
    }
}
void save_model(FILE *vocab_vec_file, FILE *class_vec_file) {
    fprintf(vocab_vec_file, "%d %d\n", vocab_size, vec_size);
    for (map<string, int>::iterator it = vocab_index.begin(); it != vocab_index.end(); ++it) {
        fprintf (vocab_vec_file, "%s\t", it->first.c_str());
        for (int i = 0; i < vec_size; ++i) {
            char sep = i == vec_size - 1 ? '\n' : '\t';
            fprintf(vocab_vec_file, "%f%c", syn0[it->second * vec_size + i], sep);
        }
    }
    char code[MAX_CODE_LEN];
    fprintf(class_vec_file, "%d\n", node_num);
    dfs_save_code(class_vec_file, 0, code, 0);
}
int main (int argc, char *argv[]) {
    FILE *train_file;
    FILE *vocab_vec_file;
    FILE *class_vec_file;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "-s") == 0) vec_size = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-m") == 0) min_word_count = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-r") == 0) max_round = atoi(argv[i + 1]);
    }
    train_file = fopen("train.dat", "r");
    vocab_vec_file = fopen("vocab_vec.dat", "w");
    class_vec_file = fopen("node_vec.dat", "w");
    if (!(train_file && vocab_vec_file && class_vec_file)) {
        printf ("Open file error\n");
        exit(1);
    }
    learn_vocab(train_file);
    init_net();
    for (int i = 0; i < max_round; ++i) {
        train_model(train_file);
    }
    save_model(vocab_vec_file, class_vec_file);
    fclose(train_file);
    fclose(vocab_vec_file);
    fclose(class_vec_file);
    return 0;
}
