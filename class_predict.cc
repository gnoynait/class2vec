#include<cstdio>
#include<string>
#include<map>
#include<cmath>
#define MAX_WORD_LEN 500
#define MAX_RECORD_LEN 50
#define MAX_CODE_LEN 50
using namespace std;
int vec_size = 100;
int class_num;
int vocab_size;
int *parent;
int *children;
float *neu;
float *syn0;
float *syn1;
map<string, int> vocab_index;
// read a word from fin
// return: 0 if '\n'
//         len if read a word
//        -1 if end of file
int read_word(FILE *fin, char *buffer) {
    if (feof(fin)) return -1;
    int len = 0;
    char ch;
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

void load_vocab(FILE *vocab_file) {
    int len = 0;
    char buffer[MAX_WORD_LEN];
    int newline = 1;
    int a = 0;
    int index = 0;
    int vocab_next_index;
    while ((len = read_word(vocab_file, buffer)) != -1) {
        if (len == 0) {
            newline = 1;
            a = 0;
        } else {
            if (newline) {
                index = vocab_next_index;
                vocab_index[buffer] = index;
                vocab_next_index++;
                newline = 0;
            } else {
                syn0[index * vec_size + a] = atof(buffer);
            }
        }
    }
}

void load_nodes(FILE *node_file) {
    char buffer[MAX_WORD_LEN];
    int index = 0;
    int len;
    int newline;
    int a;
    int node_next_index;
    while ((len = read_word(node_file, buffer)) != -1) {
        if (len == 0) {
            newline = 1;
            a = 0;
        } else {
            if (newline) {
                int prev = 0;
                for (int i = 0; i < len - 1; i++) {
                    if (buffer[i] == '/') {
                        prev = 0;
                        index = 0;
                    } else if (buffer[i] == '0') {
                        prev = index;
                        index = children[prev * 2];
                    } else {
                        prev = index;
                        index = children[prev * 2 + 1];
                    }
                }
                index = node_next_index++;
                if (buffer[len - 1] == '0') {
                    children[prev * 2] = index;
                    parent[index] = prev;
                } else {
                    children[prev * 2 + 1] = index;
                    parent[index] = prev;
                }
                a = 0;
            } else {
                syn1[index * vec_size + a] = atof(buffer);
            }
        }
    }
}

void predict(int *words, int len, char *code) {
    int level = 0;
    for (int i = 0; i < vec_size; ++i) {
        neu[i] = 0;
    }
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < vec_size; ++j) {
            neu[j] += syn0[i * vec_size + j];
        }
    }
    int node = 0;
    while (1) {
        float f = 0;
        for (int i = 0; i < vec_size; ++i) {
            f += syn1[node * vec_size + i] * neu[i];
        }
        if (1.0 / (1 + exp(-f)) < 0.5) {
            code[level++] = '0';
        } else {
            code[level++] = '1';
        }
        if (children[node * 2] == 0) break;
    }
    code[level] = '\0';
}

void process(FILE *target_file, FILE *result_file) {
    int words[MAX_RECORD_LEN];
    char buffer[MAX_WORD_LEN];
    int len_record = 0;
    char code[MAX_CODE_LEN];
    int len = 0;
    int newline = 1;
    while (1) {
        len = read_word(target_file, buffer);
        if (len > 0) {
            if (vocab_index.count(buffer)) {
                words[len_record++] = vocab_index[buffer];
            }
        } else if (len == 0) {
            predict(words, len_record, code);
            fprintf (result_file, "%s\n", code);
            len_record = 0;
        } else {
            break;
        }
    }

}



int main() {
    FILE *vocab_vec_file;
    FILE *node_vec_file;
    FILE *target_file;
    FILE *result_file;
    vocab_vec_file = fopen("vocab_vec_file.dat", "r");
    node_vec_file = fopen("node_vec_file.dat", "r");
    target_file = fopen("target_file.dat", "r");
    result_file = fopen("result.dat", "w");
    if (!(vocab_vec_file && node_vec_file && target_file && result_file)) {
        printf("open file error\n");
        exit(1);
    }
    parent = new int[class_num -1]();
    children = new int[(class_num - 1) * 2]();
    syn0 = new float[vocab_size * vec_size]();
    syn1 = new float[(class_num - 1) * vec_size]();
    neu = new float[vec_size]();
    if (!(parent && children && syn0 && syn1 && neu)) {
        printf ("memory allocation error\n");
        exit(1);
    }
    load_vocab(vocab_vec_file);
    load_nodes(node_vec_file);
    process(target_file, result_file);
    return 0;
}

