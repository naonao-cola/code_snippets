#ifndef __CRC16_H
#define __CRC16_H

unsigned short int CRC16_CCITT(unsigned char *puchMsg, unsigned int usDataLen);
unsigned short int CRC16_CCITT_FALSE(unsigned char *puchMsg, unsigned int usDataLen);
unsigned short int CRC16_XMODEM(unsigned char *puchMsg, unsigned int usDataLen);
unsigned short int CRC16_X25(unsigned char *puchMsg, unsigned int usDataLen);
unsigned short int CRC16_MODBUS(unsigned char *puchMsg, unsigned int usDataLen);
unsigned short int CRC16_IBM(unsigned char *puchMsg, unsigned int usDataLen);
unsigned short int CRC16_MAXIM(unsigned char *puchMsg, unsigned int usDataLen);
unsigned short int CRC16_USB(unsigned char *puchMsg, unsigned int usDataLen);


#endif

