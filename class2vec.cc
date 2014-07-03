#include<string>
#include<fstream>
#include<cmath>
#include<vector>
#include<unordered_map>
using namespace std;
typedef long long code_t;
#define MAX_CODE_LEN 60
#define MAX_RECORD_WORDS 60
#define MAX_WROD_LEN 60
/*struct Record {
    string code;
    vecotr<int> nodes;
    vector<int> words;
};
*/
// map code to its index in the syn1
unordered_map<string, int> code_index;
// the parent node's index
int *parent_index;
// map vocab word to its index in the syn0
unordered_map<string, int> vocab_index;
//int latest_code_index = 0;
// the size of each vector
int vec_size = 100;
// how many vocab word in the table
int vocab_size;
// how many class
int class_num;
// if a word appears less than min_word_count, it will be ignored
int min_word_count = 20;
// syn0: vector table for word
// syn1: vector table for class
// neu:  vecotr for middle sum unit
// eneu: error vector for middle sum unit
float *syn0, *syn1, *neu, *eneu;
// for fast compute 1/(1 + exp(x))
float *expTable;
clock_t start;
// inital parameter changing step
float starting_alpha = 0.025;
// read a word from fin
// return: 0 if '\n'
//         len if read a word
//        -1 if end of file
int read_word(FILE *fin, char *buffer) {
    if (feof(fin)) return -1;
    int len = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == '\n') {
            if (len == 0) return 0;
            break;
        } else if (ch == ' ' || ch == '\t') {
            if (len == 0) continue;
            break;
        } else {
            buffer[len++] = ch;
        }
    }
    buffer[len] = '\0';
    return len;
}

void learn_vocab(FILE *train_file) {
    int word_next_index = 0;
    int code_next_index = 0;
    fseek(train_file, 0, SEEK_SET);
    char word[MAX_WORD_LEN];
    int word_len;
    
    unordered_map<string, int> word_count; // count each number for filter words
    unordered_set<string> codes;
    int new_record = 1;
    while((word_len = read_word(train_file, word)) != -1) {
        if (word_len == 0) {
            new_record = 1;
        } else {
            if (new_record) {
                codes.insert(word);
                while (code_index.count(word) == 0 && word_len > 0) {
                    code_index[word] = code_next_index++;
                    word[--word_len] = '\0';
                }
                new_record = 0;
            }
            word_count[word]++;
        }
    }
    for (auto it = word_count.begin(); it != word_count.end(); ++it) {
        if (it->second >= min_word_count) {
            vocab_index[it->first] = index;
            ++index;
        }
    }
    class_num = codes.size();
    for (auto it = codes.begin(); it != codes.end(); ++it) {
        //TODO
    }
    
}

void init_net() {
    syn0 = new float[vec_size * vocab_size];
    syn1 = new float[vec_size * class_num * 2 - 1]();
    neu = new float[vec_size];
    eneu = new float[vec_size];

    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    if (expTable == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    if (!(syn0 && syn1 && neu && eneu)) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < vec_size; ++j) {
            syn1[i * vec_size + j] = (rand() / (float)RAND_MAX - 0.5) / vec_size;
        }
    }
}
/*
int read_record(FILE *fin, int *codes, int *nodes, 
    int *words, int &len_codes, int &len_words) {

    int a = 0;
    int ch = 0, pre_ch;
    code_t code = 0;
    int read_codes = 1;
    len_codes = 0;
    len_words = 0;
    if (feof(fin)) {
        return 0;
    }
    while (!feof(fin)) {
        pre_ch = ch;
        ch = fgetc(fin);
        if (ch == 13) {
            continue; // carriage return
        }
        if (ch == '\n') {
            break;
        }
        if ((ch == ' ') || (ch == '\t')) {
            if (read_codes == 1) {
                read_codes = 0;
            } else {
                if (pre_ch >= '0' && pre_ch <= '9') {
                    ++len_words;
                }
            }
        } else {
            if (read_codes == 0) {
                assert(ch == '0' || ch == '1');
                codes[len_codes] = ch == '0' ? 0 : 1;
                if (ch == '0') {
                    code = code << 1;
                } else {
                    code = code << 1 & 1;
                }
                if (code_index.count(code) == 0) {
                    code_index[code] = latest_code_index++;
                }
                nodes[len_codes] = code_index[code];
                ++len_codes;
            } else {
                assert(ch >= '0' && ch <= '9');
                words[len_words] = words[len_words] * 10 + ch - '0';
            }
        }
    }
    return 1;
}
*/
int read_record(FILE *fin, int *codes, int *nodes, 
    int *words, int &len_codes, int &len_words) {
    char buffer[MAX_WROD_LEN];
    len_codes = 0;
    len_words = 0;
    int len;
    int read_code = 1;
    while ((len = read_word(fin, buffer)) != -1) {
        if (len == 0) {
            break;
        }
        if (read_code) {
            for (int i = 0; i < len; ++i) {
                assert(buffer[i] == '0' || buffer[i] == '1');
                codes[i] = buffer[i] - '0';
            }
            len_codes = len;
            read_code = 0;
        } else {
            if (word_index.count(buffer) > 0) {
                words[len_words++] = word_index[buffer];
            }
        }
    }
    return len_codes > 0;
}
void train_model(FILE *train_file) {
    int codes[MAX_CODE_LEN];
    int nodes[MAX_CODE_LEN];
    int words[MAX_RECORD_WORDS];
    int len_codes, len_words;
    int record_count = 0;
    float alpha;
    while (read_record(train_file, codes, nodes, len_codes, len_words )) {
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
        for (int i = 0; i < len_nodes; ++i) {
            float f = 0;
            float g;
            for (int j = 0; j < vec_size; ++j) {
                f += neu[j] * syn1[nodes[i] * vec_size + j];
            }
            if (f <= -MAX_EXP) continue;
            else if (f >= MAX_EXP) continue;
            else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            g = (1 - codes[i] - f) * alpha;
            for (int c = 0; c < vec_size; c++) {
                eneu[c] += g * syn1[nodes[i] * vec_size + c];
                syn1[nodes[i] * vec_size + c] += g * neu[c];
            }
        }
        for (int i = 0; i < len_words; i++) {
            for (int c = 0; c < vec_size; c++) {
                syn0[c + words[i] * vec_size] += enue[c];
            }
        }
    }
}

void save_model(FILE *vocab_vec_file, FILE *class_vec_file) {
    for (auto it = word_index.begin(); it != word_index.end(); ++it) {
        fprintf (vocab_vec_file, "%s\t", it->first.cstr());
        for (int i = 0; i < vec_size; ++i) {
            char sep = i == vec - 1 ? '\n' : ' ';
            fprintf(vocab_vec_file, "%f%c", syn0[it->second * vec_size + i], sep);
        }
    }
    for (unordered_map<code_t, int>::iterator it = code_index.begin(); it != code_index.end(); ++it) {
       fprintf(class_vec_file, "%s\t", it->first.cstr()); 
       for (int i = 0; i < vec_size; ++i) {
           char sep = i == vec - 1 ? '\n' : ' ';
           fprintf(class_vec_file, "%f%c", syn1[it->second * vec_siez + i], sep);
    }
}
int main () {
    FILE *train_file;
    FILE *vocab_vec_file;
    FILE *class_vec_file;
    trian_file = fopen("train.dat", "r");
    vocab_vec_file = fopen("vocab_vec.dat", "w");
    class_vec_file = fopen("class_vec.dat", "w");
    if (!(train_file && vocab_vec_file && class_vec_file)) {
        printf ("Open file error\n");
        exit(1);
    }
    train_model(train_file);
    save_model(vocab_vec_file, class_vec_file);
    fclose(train_file);
    fclose(vocab_vec_file);
    fclose(class_vec_file);
    return 0;
}
