
#ifndef ET199_API_H
#define ET199_API_H


#ifdef __cplusplus
extern "C" {
#endif

int open_device(int index);
void close_device(int index);
int get_hardware_id(int index, unsigned char *bid);
int write_sign(int index, unsigned char *user_pin, unsigned char *sign, int size);
int read_sign(int index, unsigned char *user_pin, unsigned char *sign, int size);
int get_atr(int index, unsigned char *atr);
int set_atr(int index, unsigned char *atr);

#ifdef __cplusplus
}
#endif
#endif

