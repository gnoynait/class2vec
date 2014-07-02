#include<string>
#include<fstream>
#include<cmath>
#include<vector>
#include<unordered_map>
using namespace std;
typedef long long code_t;
#define MAX_CODE_LEN 60
#define MAX_RECORD_WORDS 60
struct Record {
    string code;
    vecotr<int> nodes;
    vector<int> words;
};
unordered_map<code_t, int> code_index;
int latest_code_index = 0;
int vec_size = 100;
int vocab_size;
int class_num;
float *syn0, *syn1, *neu, *eneu;
float *expTable;
clock_t start;
float starting_alpha = 0.025;
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
